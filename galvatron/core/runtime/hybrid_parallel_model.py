from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed import fsdp

from .comm_groups import gen_comm_groups
from .hybrid_parallel_config import (
    check_hp_config,
    get_chunks,
    hp_config_whole_model,
    layer_shapes_dtypes_whole_model,
    mixed_precision_dtype,
)
from galvatron.core.runtime.models.builder import build_sequential_from_arch
from .initialize import init_empty_weights
from .parallel import wrap_modules_relocation
from .pipeline.grad_reduce import _finalize_params_bf16, _register_post_backward_hook_bf16
from galvatron.core.runtime.utils.utils import get_layernorm_offset, print_rank_0, rgetattr

from galvatron.core.runtime.tensor_parallel.random import set_seed_with_group
from galvatron.core.runtime.args_schema import GalvatronRuntimeArgs
from galvatron.core.runtime.models.arch import ModelInfo, BlockNames
from galvatron.core.runtime.pipeline import PipelineParallel
from galvatron.core.runtime import parallel_state

def _tie_embedding_lm_head(hp_model, args: GalvatronRuntimeArgs):
    """Share lm_head weights with embed_tokens when untie_embeddings=False (Megatron-style)."""
    if args.model.untie_embeddings_and_output_weights or hp_model.group_size != 1:
        return
    embed_weight = rgetattr(hp_model.model_cur_stage[0].module, "embed_tokens.weight")
    lm_head = rgetattr(hp_model.model_cur_stage[-1].module, "lm_head")
    lm_head.weight = embed_weight


version_str = torch.__version__
version_major, version_minor, _ = version_str.split(".")
version_major, version_minor = int(version_major), int(version_minor)
if version_major > 1:
    if version_minor > 0:
        from torch.distributed.fsdp._runtime_utils import _register_post_backward_hook

    else:
        from torch.distributed.fsdp._runtime_utils import _register_post_backward_hooks
else:
    assert False, f"PyTorch version must be greater than 2.0, but found {torch.__version__}"


class GalvatronModel(nn.Module):
    def __init__(self, hp_model: PipelineParallel):
        super().__init__()
        from galvatron.core.runtime.parallel_state import get_args

        self.args: GalvatronRuntimeArgs = get_args()
        self.model = hp_model
        self.iter = 0

    def forward_backward(self, batch, iter=None, profiler=None, loss_func=None, **kwargs):
        args, model = self.args, self.model
        self.iter = iter if iter is not None else self.iter
        if loss_func is not None:
            if len(batch) == 1 and isinstance(batch[0], Tensor):
                batch = [batch, [self.fake_tensor(batch[0])]]
            assert (
                isinstance(batch, (tuple, list))
                and isinstance(batch[0], (tuple, list))
                and isinstance(batch[1], (tuple, list))
            )
        else:
            loss_func = self.fake_loss_func
            assert isinstance(batch, (tuple, list))
            batch = [batch, [self.fake_tensor(batch[0])]]
        if args.parallel.pp_deg > 1:
            if args.parallel.pipeline_type == "gpipe":
                loss = model.gpipe_forward(batch, loss_func, **kwargs)
                if profiler is not None:
                    profiler.profile_memory(self.iter, "After Forward")
                model.gpipe_backward()
            elif args.parallel.pipeline_type == "pipedream_flush":
                loss = model.pipedream_flush_forward_backward(batch, loss_func, **kwargs)
        else:
                loss = model.no_pipeline_forward_backward(
                batch, loss_func, forward_only=args.profile.profile_forward, profiler=profiler, iter=self.iter, **kwargs
            )
        self.iter += 1
        return self.loss_to_cpu(loss)

    def fake_tensor(self, x):
        return torch.zeros([x.shape[0], 1], dtype=x.dtype, device=x.device)

    def fake_loss_func(self, labels, outputs):
        if torch.numel(outputs[0]) > 1:
            loss = outputs[0].mean()
            return loss, loss.clone().detach()
        return outputs[0], outputs[0].clone().detach()

    def loss_to_cpu(self, loss):
        if isinstance(loss, (list, tuple)):  # Average loss of each microbatch
            if len(loss) == 0:
                return None
            loss = np.mean([l.item() for l in loss])
        else:
            loss = loss.item()
        return loss

def construct_hybrid_parallel_model_api(
    arch_list: List[str],
    args:GalvatronRuntimeArgs,
    hybrid_parallel_configs:dict,
    model_info:ModelInfo,
    block_names:BlockNames,
    layernorm_name: Optional[List[str]] = None,
    tied_wte_attr_names=None,
    load_module_func=None,
    meta_init_buffer=True,
) -> GalvatronModel:
    """Build a hybrid-parallel model from an architecture list.

    Args:
        arch_list: Module type sequence, e.g.
            ``["embedding", "decoder", "decoder", ..., "prenorm", "lm_head"]``.
        args: Galvatron args (with ``args.model``, ``args.train``, ``args.parallel``).
        hybrid_parallel_configs: From ``get_hybrid_parallel_configs_api``.
        layernorm_name: Substrings used to find LayerNorm modules for SP allreduce.
            ``None`` = auto (covers common names).
        tied_wte_attr_names: Attribute names for weight-tied embedding / lm_head.
        load_module_func: Optional checkpoint loading callback.
        meta_init_buffer: Whether to init buffers on meta device.
    """

    hp_configs = hybrid_parallel_configs

    if args.parallel.mixed_precision == "bf16":
        assert version_major > 1 and version_minor > 0, "Mixed precision training is only supported for torch > 2.0.1"
        fsdp._runtime_utils._register_post_backward_hook = _register_post_backward_hook_bf16
        fsdp._runtime_utils._finalize_params = _finalize_params_bf16
    # Get model-specific model info: module_types, layernum_list, layer_shapes_list, layer_dtypes_list
    module_types = model_info.module_types()
    layernum_list = model_info.layernums()
    layer_shapes_list = model_info.shapes()
    layer_dtypes_list = model_info.dtypes()

    # Check the validity of hp_configs
    check_hp_config(hp_configs, layernum_list)

    # Calculate shapes and dtypes for whole model (including embed/cls/... layers)
    shapes_whole, dtypes_whole = layer_shapes_dtypes_whole_model(
        module_types, layernum_list, layer_shapes_list, layer_dtypes_list
    )

    # Get hp_configs_whole for the whole model (including embed/cls/... layers)
    hp_configs_whole = hp_config_whole_model(
        module_types, hp_configs,
        vocab_sdp=args.parallel.vocab_sdp,
        embed_ckpt=0,
        vocab_tp=args.parallel.vocab_tp,
        vocab_sp=args.parallel.vocab_sp,
        vocab_cp=args.parallel.vocab_cp,
    )

    # [Step 0] Generate communication groups
    print_rank_0("Generating communication groups...")
    (
        pp_group,
        tp_groups_whole,
        sp_groups_whole,
        cp_groups_whole,
        dp_groups_whole,
        seq_data_groups_whole,
        ep_groups_whole,
        tp_of_ep_groups_whole,
        tp_and_ep_groups_whole,
        dp_of_ep_groups_whole,
        allgather_cp_groups_whole,
        split_cp_groups_whole,
        allgather_tp_sp_cp_groups_whole,
        split_tp_sp_cp_groups_whole,
        fused_allgather_groups_whole,
        fused_split_groups_whole,
        embedding_group,
    ) = gen_comm_groups(
        hp_configs_whole["tp_sizes_whole"],
        hp_configs_whole["sp_sizes_whole"],
        hp_configs_whole["cp_sizes_whole"],
        hp_configs_whole["ep_sizes_whole"],
        hp_configs_whole["tp_of_ep_sizes_whole"],
        hp_configs_whole["pp_deg"],
        is_moe_model=hp_configs_whole["is_moe_model"],
        show_rank=0,
    )

    parallel_state.set_pp_comm_group(pp_group)

    parallel_state.set_vocab_tp_sp_comm_group(sp_groups_whole[0] if args.parallel.use_ulysses else tp_groups_whole[0])
    parallel_state.set_vocab_cp_comm_group(cp_groups_whole[0])
    parallel_state.set_vocab_dp_comm_group(dp_groups_whole[0])
    parallel_state.set_vocab_tp_sp_src_rank(sp_groups_whole[0].ranks[0] if args.parallel.use_ulysses else tp_groups_whole[0].ranks[0])

    parallel_state.set_tp_whole_comm_group(tp_groups_whole[1:-2])
    parallel_state.set_sp_whole_comm_group(sp_groups_whole[1:-2])
    parallel_state.set_dp_whole_comm_group(dp_groups_whole[1:-2])
    parallel_state.set_cp_whole_comm_group(cp_groups_whole[1:-2])
    parallel_state.set_sdp_whole_comm_group(seq_data_groups_whole[1:-2])

    assert args.model.shape_order == "SBH", "Shape order must be SBH for hybrid parallel model!"

    set_seed_with_group(
        tp_groups=tp_groups_whole,
        tp_and_ep_groups=tp_and_ep_groups_whole,
    )

    # [Step 1 - 2] Construct TP & Sequantial model using model-specific sequential function
    print_rank_0("Constructing TP & Sequantial model using model-specific sequential function...")
    if args.model.initialize_on_meta:
        with init_empty_weights(meta_init_buffer):
            model = build_sequential_from_arch(
                arch_list, args, 
                tp_groups_whole, 
                sp_groups_whole, 
                cp_groups_whole,
                ep_groups_whole,
                tp_of_ep_groups_whole,
                tp_and_ep_groups_whole,
            )
    else:
        model = build_sequential_from_arch(
            arch_list, args, 
            tp_groups_whole, 
            sp_groups_whole, 
            cp_groups_whole,
            ep_groups_whole,
            tp_of_ep_groups_whole,
            tp_and_ep_groups_whole,
        )

    # [Step 3] Wrap Relocation modules if necessary
    print_rank_0("Wrapping Relocation modules if necessary...")
    model = wrap_modules_relocation(
        model, allgather_cp_groups_whole, allgather_tp_sp_cp_groups_whole,
        split_cp_groups_whole, split_tp_sp_cp_groups_whole,
        fused_allgather_groups_whole, fused_split_groups_whole,
    )
    ln_offset, ln_size = get_layernorm_offset(model, layernorm_name)
    assert len(ln_offset) == len(dp_groups_whole)

    # [Step 4] Construct Pipeline Module and place the layers on corresponding devices
    from galvatron.core.runtime.pipeline import PipelineParallel
    print_rank_0("Constructing Pipeline Module and placing the layers on corresponding devices...")
    hp_model = PipelineParallel(
        model=model,
        model_ranks=hp_configs_whole["pp_ranks_whole"],
        layer_output_tensor_shapes=shapes_whole,
        layer_output_tensor_dtypes=dtypes_whole,
        layer_dp_sizes=hp_configs_whole["dp_sizes_whole"],
        layer_tp_sizes=hp_configs_whole["tp_sizes_whole"],
        layer_sp_sizes=hp_configs_whole["sp_sizes_whole"],
        layer_cp_sizes=hp_configs_whole["cp_sizes_whole"],
        chunks=get_chunks(args),
        process_group=pp_group,
        embedding_group=embedding_group,
        nproc_per_node=8,
        info=False,
        tied_wte_attr_names=tied_wte_attr_names,
    )

    # [Step 5] Wrap Data Parallel modules based on dp_types & dp_groups
    hp_model.wrap_pipeline_modules_data_parallel(
        hp_configs_whole["dp_types_whole"],
        seq_data_groups_whole,
        module_types=module_types,
        dp_of_ep_groups=dp_of_ep_groups_whole,
        mixed_precision=mixed_precision_dtype(args.parallel.mixed_precision),
        wrap_block_name=block_names.wrap_block_name,
        wrap_other_block_name=block_names.wrap_other_block_name,
        tp_groups=tp_groups_whole,
        tp_of_ep_groups=tp_of_ep_groups_whole,
        ep_groups=ep_groups_whole,
        all_block_name=block_names.all_block_name,
        load_module_func=load_module_func,
    )

    _tie_embedding_lm_head(hp_model, args)

    hp_model.gen_sp_layernorm_info(
        layer_module_types=module_types,
        layer_tp_groups=tp_groups_whole,
        ln_offset=ln_offset,
        ln_size=ln_size,
        all_block_name=block_names.all_block_name,
    )

    # [Step 6] Wrap checkpoint based on checkpoint_flags
    print_rank_0("Wrapping checkpoint based on checkpoint_flags...")
    hp_model.wrap_pipeline_modules_checkpoint(
        hp_configs_whole["checkpoint_flags_whole"], wrap_block_name=block_names.wrap_checkpoint_block_name
    )

    model = GalvatronModel(hp_model)
    model.dp_groups_whole = dp_groups_whole
    model.tp_groups_whole = tp_groups_whole
    model.sp_groups_whole = sp_groups_whole
    model.cp_groups_whole = cp_groups_whole
    model.sdp_groups_whole = seq_data_groups_whole
    model.ep_groups_whole = ep_groups_whole
    model.tp_of_ep_groups_whole = tp_of_ep_groups_whole
    model.tp_and_ep_groups_whole = tp_and_ep_groups_whole
    model.dp_of_ep_groups_whole = dp_of_ep_groups_whole
    model.hybrid_parallel_configs = hybrid_parallel_configs

    return model
