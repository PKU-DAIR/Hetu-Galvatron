import json
import os

import numpy as np
import torch

from galvatron.utils import config2strategy, read_json_config, str2array


def get_pp_ranks_enc(pp_divide):
    pp_ranks_enc = []
    pp_deg = len(pp_divide)
    for i in range(pp_deg):
        pp_ranks_enc += [i] * pp_divide[i]
    return pp_ranks_enc


def get_hybrid_parallel_configs_api(config, args, model_info):
    local_rank = args.local_rank
    world_size = torch.distributed.get_world_size()
    config_type = "JSON" if args.galvatron_config_path not in [None, "None"] else "GLOBAL"
    layernum_list = model_info(config, args).layernums()
    total_layer_num = sum(layernum_list)
    if local_rank == 0:
        print("======================== Galvatron Parallel Config =============================")
        print("Galvatron parallel config mode: [%s config mode]" % config_type)
    if config_type == "GLOBAL":
        pp_deg = args.pp_deg
        tp_sizes_enc = [args.global_tp_deg] * total_layer_num if args.global_tp_deg > 0 else [1] * total_layer_num
        tp_consecutive_flags = (
            [args.global_tp_consec] * total_layer_num if args.global_tp_consec in [0, 1] else [1] * total_layer_num
        )
        dp_types_enc = total_layer_num * [args.sdp]
        checkpoint_flags_enc = [args.global_checkpoint] * total_layer_num
        pp_divide = None
        if args.use_ulysses:
            args.vocab_sp = 1
            use_sp = [1] * total_layer_num
        else:
            args.vocab_sp = 0
            use_sp = [0] * total_layer_num
    else:
        if isinstance(args.galvatron_config_path, str):
            galvatron_config = read_json_config(args.galvatron_config_path)
        else:
            galvatron_config = args.galvatron_config_path
        pp_deg, tp_sizes_enc, tp_consecutive_flags, dp_types_enc, use_sp, vtp, vsp = config2strategy(galvatron_config)
        bsz, chunks = galvatron_config["global_bsz"], galvatron_config["chunks"]
        checkpoint_flags_enc = (
            str2array(galvatron_config["checkpoint"])
            if "checkpoint" in galvatron_config.keys()
            else [0] * len(tp_sizes_enc)
        )
        pp_divide = str2array(galvatron_config["pp_division"]) if "pp_division" in galvatron_config.keys() else None
        if isinstance(args.galvatron_config_path, str):
            config_source = "Galvatron JSON config %s" % args.galvatron_config_path
        else:
            config_source = "Galvatron JSON config"
        args.pipeline_type = (
            galvatron_config["pipeline_type"] if "pipeline_type" in galvatron_config.keys() else args.pipeline_type
        )
        args.default_dp_type = (
            galvatron_config["default_dp_type"]
            if "default_dp_type" in galvatron_config.keys()
            else args.default_dp_type
        )
        args.embed_sdp = galvatron_config["embed_sdp"] if "embed_sdp" in galvatron_config.keys() else args.embed_sdp
        if local_rank == 0 and (
            total_layer_num != len(tp_sizes_enc) or args.chunks != chunks or args.global_train_batch_size != bsz
        ):
            print("[Notice] The following hyper-parameters will be overwritten by Galvatron %s config:" % config_type)
            if args.global_train_batch_size != bsz:
                print("   global_batch_size =", bsz)
            if args.chunks != chunks:
                print("   chunks =", chunks)
            if total_layer_num != len(tp_sizes_enc):
                assert (False, "Layer_num in json config does not match layer_num in the model!")
        args.global_train_batch_size = bsz
        args.chunks = chunks
        args.pp_deg = pp_deg
        args.vocab_tp = vtp
        args.vocab_sp = vsp

    if pp_divide is None:
        avg_layer_num = total_layer_num // pp_deg
        last_layer_num = total_layer_num - avg_layer_num * (pp_deg - 1)
        pp_divide = [avg_layer_num] * (pp_deg - 1) + [last_layer_num]
    pp_ranks_enc = get_pp_ranks_enc(pp_divide)

    min_tp = min(min(tp_sizes_enc), args.vocab_tp)
    assert (
        args.global_train_batch_size % (world_size // pp_deg // min_tp) == 0
    ), "global_train_batch_size should be multiple of world_size//pp_deg!"
    hybrid_parallel_configs = {
        "pp_deg": pp_deg,
        "tp_sizes_enc": tp_sizes_enc,
        "tp_consecutive_flags": tp_consecutive_flags,
        "dp_types_enc": dp_types_enc,
        "checkpoint_flags_enc": checkpoint_flags_enc,
        "pp_ranks_enc": pp_ranks_enc,
        "pp_division": pp_divide,
        "use_sp": use_sp,
        "vocab_tp": args.vocab_tp,
        "vocab_sp": args.vocab_sp,
        "default_dp_type": args.default_dp_type,
        "global_train_batch_size": args.global_train_batch_size,
    }

    if args.distributed_checkpoint:
        json_path = os.path.join(args.load, f"hybrid_parallel_configs.json")
        checkponit_hybrid_parallel_configs = json.load(open(json_path, "r"))
        assert (
            hybrid_parallel_configs.keys() == checkponit_hybrid_parallel_configs.keys()
        ), "Hybrid parallel configs are not equal, %s vs %s" % (
            hybrid_parallel_configs.keys(),
            checkponit_hybrid_parallel_configs.keys(),
        )
        for key in hybrid_parallel_configs.keys():
            assert (
                hybrid_parallel_configs[key] == checkponit_hybrid_parallel_configs[key]
            ), f"Hybrid parallel configs are not equal for key {key}, {hybrid_parallel_configs[key]} vs {checkponit_hybrid_parallel_configs[key]}"

    if local_rank == 0:
        if config_type == "GLOBAL":
            print("[GLOBAL config mode] Loaded global hybrid parallel strategy:")
            dp_type = "sdp" if args.sdp else "dp"
            tp_deg, tp_consec = tp_sizes_enc[0], tp_consecutive_flags[0]
            dp_deg = world_size // args.global_tp_deg // args.pp_deg
            print("   global_batch_size: %d, chunks: %d" % (args.global_train_batch_size, get_chunks(args)))
            print(
                "   pp_deg: %d, tp_deg: %d, %s_deg: %d, tp_consecutive_flag: %d, checkpoint_flag: %d"
                % (pp_deg, tp_deg, dp_type, dp_deg, tp_consec, args.global_checkpoint)
            )
            embed_sdp = ", embed_sdp: 1" if args.embed_sdp else ""
            print(
                "   pipeline_type: %s, default_dp_type: %s, dtype: %s%s"
                % (args.pipeline_type, args.default_dp_type, args.mixed_precision, embed_sdp)
            )
            print_hp_config("pp_division", pp_divide)
            print_hp_config("pp_ranks", pp_ranks_enc)
            print_hp_config("use_sp", [args.use_ulysses])
            print("================================================================================")
        else:
            print("[%s config mode] Loaded hybrid parallel config from %s:" % (config_type, config_source))
            print(
                "   global_batch_size: %d, chunks: %d, pp_deg: %d" % (args.global_train_batch_size, args.chunks, pp_deg)
            )
            embed_sdp = ", embed_sdp: 1" if args.embed_sdp else ""
            print(
                "   pipeline_type: %s, default_dp_type: %s, dtype: %s%s"
                % (args.pipeline_type, args.default_dp_type, args.mixed_precision, embed_sdp)
            )
            print_hp_configs(hybrid_parallel_configs)
    return hybrid_parallel_configs


class ModelInfo:
    def __init__(self):
        return

    def set_layernums(self, info):
        self.layernum_list = info

    def set_shapes(self, info):
        self.layer_shapes_list = info

    def set_dtypes(self, info):
        self.layer_dtypes_list = info

    def set_module_types(self, info):
        self.layer_module_types = info

    def layernums(self):
        return self.layernum_list

    def shapes(self):
        return self.layer_shapes_list

    def dtypes(self):
        return self.layer_dtypes_list

    def module_types(self):
        return self.layer_module_types


def check_hp_config(hp_configs, layernum_list):
    pp_deg, tp_sizes_enc, tp_consecutive_flags, dp_types_enc, pp_ranks_enc, checkpoint_flags_enc = (
        hp_configs["pp_deg"],
        hp_configs["tp_sizes_enc"],
        hp_configs["tp_consecutive_flags"],
        hp_configs["dp_types_enc"],
        hp_configs["pp_ranks_enc"],
        hp_configs["checkpoint_flags_enc"],
    )
    total_layer_num = sum(layernum_list)
    assert total_layer_num == len(tp_sizes_enc)
    assert total_layer_num == len(tp_consecutive_flags)
    assert total_layer_num == len(dp_types_enc)
    assert total_layer_num == len(pp_ranks_enc)
    assert total_layer_num == len(checkpoint_flags_enc)
    world_size = torch.distributed.get_world_size()
    for tp_size in tp_sizes_enc:
        assert (
            tp_size <= world_size // pp_deg and (world_size // pp_deg) % tp_size == 0 and tp_size >= 1
        ), "Wrong tp_size!"
    for tp_consec in tp_consecutive_flags:
        assert tp_consec == 0 or tp_consec == 1, "Wrong tp_consec!"
    for dp_type in dp_types_enc:
        assert dp_type == 0 or dp_type == 1 or dp_type is None, "Wrong dp_type!"
    for pp_rank in pp_ranks_enc:
        assert pp_rank >= 0 and pp_rank <= pp_deg - 1, "Wrong pp_rank!"
    for ckpt in checkpoint_flags_enc:
        assert ckpt == 0 or ckpt == 1 or ckpt == 2, "Wrong checkpoint_flag!"


def print_hp_config(key, val):
    if isinstance(val, (list, tuple)):
        padding = 28 - len(key) if 28 - len(key) > 0 else 0
        name = "   " + key + ":" + padding * " "
        print(name, val)


def print_hp_configs(hp_configs):
    for key, val in hp_configs.items():
        print_hp_config(key, val)
    print("================================================================================")


def hp_config_whole_model(module_types, hp_configs, embed_sdp=0, embed_ckpt=0, vocab_tp=1, vocab_sp=0):
    pp_deg, tp_sizes_enc, use_sp, tp_consecutive_flags, dp_types_enc, pp_ranks_enc, checkpoint_flags_enc = (
        hp_configs["pp_deg"],
        hp_configs["tp_sizes_enc"],
        hp_configs["use_sp"],
        hp_configs["tp_consecutive_flags"],
        hp_configs["dp_types_enc"],
        hp_configs["pp_ranks_enc"],
        hp_configs["checkpoint_flags_enc"],
    )

    hp_configs_whole = dict()
    hp_configs_whole["pp_deg"] = hp_configs["pp_deg"]
    keys = [
        "tp_sizes_whole",
        "sp_sizes_whole",
        "tp_consec_whole",
        "dp_types_whole",
        "pp_ranks_whole",
        "checkpoint_flags_whole",
    ]
    for key in keys:
        hp_configs_whole[key] = []

    idx_enc = 0
    for module_type in module_types:
        if module_type[-3:] == "enc" or module_type[-3:] == "dec":
            if use_sp[idx_enc] == 1:
                hp_configs_whole["sp_sizes_whole"].append(tp_sizes_enc[idx_enc])
                hp_configs_whole["tp_sizes_whole"].append(1)
            else:
                hp_configs_whole["tp_sizes_whole"].append(tp_sizes_enc[idx_enc])
                hp_configs_whole["sp_sizes_whole"].append(1)
            hp_configs_whole["dp_types_whole"].append(dp_types_enc[idx_enc])
            hp_configs_whole["pp_ranks_whole"].append(pp_ranks_enc[idx_enc])
            hp_configs_whole["tp_consec_whole"].append(tp_consecutive_flags[idx_enc])
            hp_configs_whole["checkpoint_flags_whole"].append(checkpoint_flags_enc[idx_enc])
            idx_enc += 1
        else:
            if vocab_sp == 1:
                hp_configs_whole["sp_sizes_whole"].append(vocab_tp)
                hp_configs_whole["tp_sizes_whole"].append(1)
            else:
                hp_configs_whole["tp_sizes_whole"].append(vocab_tp)
                hp_configs_whole["sp_sizes_whole"].append(1)
            hp_configs_whole["dp_types_whole"].append(embed_sdp)
            hp_configs_whole["pp_ranks_whole"].append(
                pp_ranks_enc[idx_enc] if idx_enc < len(pp_ranks_enc) else pp_ranks_enc[-1]
            )
            hp_configs_whole["tp_consec_whole"].append(1)
            hp_configs_whole["checkpoint_flags_whole"].append(embed_ckpt)

    world_size = torch.distributed.get_world_size()
    hp_configs_whole["dp_sizes_whole"] = [
        world_size // pp_deg // tp_size // sp_size
        for tp_size, sp_size in zip(hp_configs_whole["tp_sizes_whole"], hp_configs_whole["sp_sizes_whole"])
    ]
    from galvatron.core import get_args

    if get_args().local_rank == 0:
        print("Model Layer Types:")
        print(module_types)
        # print_hp_configs(hp_configs)
        print_hp_configs(hp_configs_whole)
        test_dict = {}
        for key in keys:
            if isinstance(hp_configs_whole[key], (list, tuple)):
                test_dict[key + "_check"] = get_enc_groups(hp_configs_whole[key], module_types)
        # print_hp_configs(test_dict)
    return hp_configs_whole


def get_enc_groups(groups_whole, module_types):
    groups = []
    assert len(groups_whole) == len(module_types)
    for i, module_type in enumerate(module_types):
        if module_type[-3:] == "enc" or module_type[-3:] == "dec":
            groups.append(groups_whole[i])
    return groups


def mixed_precision_dtype(mixed_precision):
    return {"fp32": torch.float, "fp16": torch.float16, "bf16": torch.bfloat16}[mixed_precision]


def layer_shapes_dtypes_whole_model(module_types, layernum_list, layer_shapes_list, layer_dtypes_list):
    assert len(layernum_list) == len(layer_shapes_list)
    assert len(layernum_list) == len(layer_dtypes_list)
    shapes_enc, dtypes_enc = [], []
    for layernum, layer_shape, layer_dtype in zip(layernum_list, layer_shapes_list, layer_dtypes_list):
        shapes_enc.extend([layer_shape] * layernum)
        dtypes_enc.extend([layer_dtype] * layernum)
    shapes_whole, dtypes_whole = [], []
    idx_enc = 0
    for module_type in module_types:
        if "enc" in module_type or "dec" in module_type:
            shapes_whole.append(shapes_enc[idx_enc])
            dtypes_whole.append(dtypes_enc[idx_enc])
            idx_enc += 1
        else:
            if idx_enc == 0 or idx_enc == len(shapes_enc):
                shapes_whole.append(None)
                dtypes_whole.append(None)
            else:
                shapes_whole.append(shapes_enc[idx_enc])
                dtypes_whole.append(dtypes_enc[idx_enc])
    # if get_args().local_rank == 0:
    #     print('Model Layer Shapes:')
    #     print(shapes_whole)
    #     print('Model Layer Dtypes:')
    #     print(dtypes_whole)
    return shapes_whole, dtypes_whole


def get_chunks(args):
    if args.chunks == -1:
        args.chunks = 1
        if args.pp_deg > 1:
            world_size = torch.distributed.get_world_size()
            max_dp_deg = world_size // args.pp_deg
            local_bsz = args.global_train_batch_size // max_dp_deg
            optimal_micro_bsz = np.ceil(local_bsz / 4)
            optimal_micro_bsz = 1 if optimal_micro_bsz == 0 else optimal_micro_bsz
            args.chunks = int(optimal_micro_bsz)
    return args.chunks
