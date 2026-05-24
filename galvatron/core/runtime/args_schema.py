"""Pydantic models for Galvatron runtime/training arguments only. Merged view: galvatron.core.args_schema."""
from typing import Literal, Optional, List, Callable

import torch
from pydantic import BaseModel, ConfigDict, Field, ImportString, field_validator

__all__ = [
    "GalvatronParallelArgs",
    "GalvatronModelArgs",
    "GalvatronProfileArgs",
    "GalvatronRuntimeArgs",
    "GalvatronTrainingArgs",
    "CommonTrainArgs",
    "CommonDataArgs",
    "CommonCkptArgs",
]

class GalvatronParallelArgs(BaseModel):
    """Parallelism and strategy."""

    pp_deg: int = Field(default=1, ge=1, description="Pipeline parallel degree.")
    global_tp_deg: int = Field(default=1, ge=1, description="Global tensor parallel degree.")
    global_tp_consec: Literal[0, 1] = Field(default=1, description="Global tensor parallel group consecutive flag.")
    global_cp_deg: int = Field(default=1, ge=1, description="Context parallel degree.")
    global_ep_deg: int = Field(default=1, ge=1, description="Experts parallel degree.")
    global_tp_of_ep_deg: int = Field(default=1, ge=1, description="Tensor parallel degree of experts.")
    global_checkpoint: int = Field(default=0, description="Global checkpoint flag.")
    cp_mode: Literal["ring", "zigzag"] = Field(default="zigzag", description="Context parallel communication mode.")
    sdp: Literal[0, 1] = Field(default=0, description="Apply SDP (zero-3).")
    default_dp_type: Literal["ddp", "zero2", "zero3"] = Field(default="ddp", description="Default data parallel type.")
    pipeline_type: Literal["gpipe", "pipedream_flush"] = Field(default="gpipe", description="Galvatron pipeline type.")
    galvatron_config_path: Optional[str] = Field(
        default=None,
        description="Galvatron strategy config path. If not None, galvatron will run according to json config file.",
    )
    vocab_sdp: Literal[0, 1] = Field(default=0, description="Apply SDP (zero-3) for Embeddings and cls.")
    vocab_tp: int = Field(default=1, ge=1, description="Tensor parallel degree of vocab.")
    vocab_cp: int = Field(default=1, ge=1, description="Context parallel degree of vocab.")
    vocab_sp: int = Field(default=1, description="Sequence parallel degree of vocab.")
    async_grad_reduce: bool = Field(
        default=True,
        description="If False, gradient will be reduced every micro batch. Ensure Zero3 memory cost when chunk > 1.",
    )
    mixed_precision: Literal["fp32", "fp16", "bf16"] = Field(default="bf16", description="Mixed precision option.")
    use_ulysses: bool = Field(default=False, description="Whether to use DeepSpeed Ulysses or Megatron-TP.")
    reduce_in_fp32: bool = Field(default=False, description="Use fp32 for gradient reduction.")
    entropy_in_fp32: bool = Field(default=False, description="Use fp32 for entropy calculation.")



class GalvatronModelArgs(BaseModel):
    """Model and training basics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hf_model_name_or_path: Optional[str] = Field(
        default=None,
        description=(
            "HuggingFace model name, path, or config class name. "
            "When set, model architecture fields (hidden_size, num_layers, normalization, ...) "
            "are auto-populated from the HF config. Manual overrides still take priority."
        ),
    )
    model_config_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to a YAML model config file (e.g. model_configs/llama2-7b.yaml). "
            "Fields in the file use the same names as GalvatronModelArgs. "
            "Null fields are skipped; non-null fields populate args.model.*. "
            "If hf_model_name_or_path is also set in the file, auto-detection runs first."
        ),
    )
    is_moe_model: bool = Field(default=False, description="Whether to use MoE.")
    set_experts_manually: int = Field(
        default=0,
        description="Whether to set experts config manually (doesn't overwrite other model configs).",
    )
    set_model_config_manually: int = Field(
        default=0,
        description="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten.",
    )
    set_layernum_manually: int = Field(
        default=0,
        description="Whether to set layernum config manually (doesn't overwrite other model configs).",
    )
    set_seqlen_manually: int = Field(
        default=0,
        description="Whether to set sequence length config manually (doesn't overwrite other model configs).",
    )
    initialize_on_meta: Literal[0, 1] = Field(default=1, description="Whether to initialize parameters on meta device.")
    # TODO: remove shape order or add bhd?
    shape_order: Literal["SBH", "BSH"] = Field(default="SBH", description="Model shape order.")
    dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout rate.")
    print_loss: int = Field(default=0, description="Whether to check model correctness.")
    model_size: Optional[str] = Field(default=None, description="Model size.")
    vocab_size: Optional[int] = Field(default=None, description="Size of vocab before EOD or padding.")
    padded_vocab_size: Optional[int] = Field(default=None, description="Size of vocab after EOD or padding.")
    hidden_size: Optional[int] = Field(default=None, description="Transformer hidden size.")
    ffn_hidden_size: Optional[int] = Field(default=None, description="Transformer intermediate size.")
    num_layers: Optional[int] = Field(default=None, description="Number of transformer layers.")
    num_attention_heads: Optional[int] = Field(default=None, description="Number of transformer attention heads.")
    num_query_groups: Optional[int] = Field(default=None, description="Number of key value heads (GQA). None = MHA (kv_heads == num_attention_heads).")
    kv_channels: Optional[int] = Field(default=None, description="Projection weights dimension in multi-head attention (head_dim).")
    attention_dropout: Optional[float] = Field(default=0.0, description="Attention dropout rate.")
    hidden_dropout: Optional[float] = Field(default=0.0, description="Hidden dropout rate.")
    add_qkv_bias: bool = Field(default=False, description="Add a bias term only for QKV projections.")
    layernorm_epsilon: Optional[float] = Field(default=1e-5, description="Epsilon for layer norm and RMS norm.")
    qk_layernorm: bool = Field(default=False, description="Apply LayerNorm/RMSNorm to Q and K projections before attention (Qwen3, Llama4, Gemma2).")
    position_embedding_type: Literal["learned_absolute", "rope", "mrope", "relative", "none"] = Field(default="rope", description="Position embedding type.")
    rotary_base: Optional[int] = Field(default=10000, description="Base to use for rotary positional embeddings.")
    rotary_percent: Optional[float] = Field(default=1.0, description="Percent of rotary dimension to use.")
    rotary_interleaved: bool = Field(default=False, description="Use interleaved rotary embedding.")
    rotary_seq_len_interpolation_factor: Optional[int] = Field(default=None, description="Sequence length interpolation factor for rotary embeddings.")
    mrope_section: Optional[List[int]] = Field(default=None, description="Multimodal rope section is for channel dimension, empty by default.")
    make_vocab_size_divisible_by: Optional[int] = Field(default=128, description="Pad the vocab size to be divisible by this value.")
    normalization: Literal["LayerNorm", "RMSNorm"] = Field(default="RMSNorm", description="Normalization technique to use.")
    norm_epsilon: Optional[float] = Field(default=1e-5, description="Epsilon for layer norm and RMS norm.")
    multi_latent_attention: bool = Field(default=False, description="Use multi-latent attention.")
    apply_rope_fusion: bool = Field(default=False, description="Apply rope fusion.")
    add_bias_linear: bool = Field(default=False, description="Include a bias term in all linear layers.")
    bias_activation_fusion: bool = Field(default=False, description="Fuse bias add into activation function (gelu/swiglu).")
    activation_func_fp8_input_store: bool = Field(default=False, description="Store MLP activation input in FP8 for backprop to save memory.")
    gated_linear_unit: bool = Field(default=True, description="Use a gated linear unit (e.g. SwiGLU) for the first MLP linear layer.")
    activation_func: ImportString[Callable] = Field(default="torch.nn.functional.gelu", description="Activation function for the MLP non-linearity.")
    untie_embeddings_and_output_weights: bool = Field(default=True, description="Untie embeddings and output weights.")

    num_moe_experts: Optional[int] = Field(default=None, description="Number of experts in MoE layer. None means no MoE.")
    moe_ffn_hidden_size: Optional[int] = Field(default=None, description="MoE FFN hidden size. Defaults to ffn_hidden_size when None.")
    # --- Router ---
    moe_router_topk: int = Field(default=2, description="Number of experts to route to for each token.")
    moe_router_load_balancing_type: Literal["none", "aux_loss", "seq_aux_loss", "sinkhorn"] = Field(default="aux_loss", description="MoE router load balancing type.")
    moe_router_score_function: Literal["softmax", "sigmoid"] = Field(default="softmax", description="Score function for MoE routing.")
    moe_router_pre_softmax: bool = Field(default=False, description="Enable pre-softmax routing (softmax before top-k selection).")
    moe_router_topk_scaling_factor: Optional[float] = Field(default=None, description="Scaling factor for routing score in top-k selection (only with pre-softmax).")
    moe_router_num_groups: Optional[int] = Field(default=None, description="Number of groups to divide experts into for group-limited routing.")
    moe_router_group_topk: Optional[int] = Field(default=None, description="Number of selected groups for group-limited routing.")
    moe_router_enable_expert_bias: bool = Field(default=False, description="TopK routing with dynamic per-expert bias (aux-loss-free load balancing).")
    moe_router_dtype: Optional[Literal["fp32", "fp64"]] = Field(default=None, description="Data type for routing computation. None means use the input dtype.")
    deterministic_mode: bool = Field(default=False, description="Whether to use deterministic mode in router top-k selection.")
    # --- Loss ---
    moe_aux_loss_coeff: float = Field(default=0.0, description="Scaling coefficient for the aux loss (e.g. 1e-2 is a good start).")
    moe_z_loss_coeff: Optional[float] = Field(default=None, description="Scaling coefficient for the z-loss (e.g. 1e-3 is a good start).")
    # --- Token dispatch ---
    moe_token_dispatcher_type: Literal["allgather", "alltoall_seq", "alltoall", "flex"] = Field(default="allgather", description="MoE token dispatcher type.")
    moe_expert_capacity_factor: Optional[float] = Field(default=None, description="Capacity factor for each expert. None means no token dropping.")
    moe_pad_expert_input_to_capacity: bool = Field(default=False, description="Pad input for each expert to match expert capacity length.")
    moe_token_drop_policy: Literal["probs", "position"] = Field(default="probs", description="Token drop policy when capacity is exceeded: 'probs' drops lowest-prob tokens, 'position' drops trailing tokens.")
    moe_input_jitter_eps: Optional[float] = Field(default=None, description="Add noise to input tensor by applying jitter with specified epsilon.")
    moe_permute_fusion: bool = Field(default=True, description="Fuse token rearrangement ops during token dispatching.")
    moe_enable_deepep: bool = Field(default=False, description="Enable DeepEP for efficient token dispatching (requires flex dispatcher).")
    # --- Shared expert ---
    moe_shared_expert_intermediate_size: Optional[int] = Field(default=None, description="Shared expert total FFN hidden size. None means no shared expert.")
    moe_shared_expert_overlap: bool = Field(default=False, description="Overlap shared expert compute with dispatcher communications (requires alltoall dispatcher).")
    # --- Misc ---
    calculate_per_token_loss: bool = Field(default=False, description="Whether to scale aux loss by number of tokens (per-token loss mode).")
    # --- MoE MLP ---
    moe_grouped_gemm: bool = Field(default=False, description="Use grouped GEMM for MoE MLP.")

    # ===== Model parallel config =====
    params_dtype: torch.dtype = Field(default=torch.float32, description="Parameters dtype.")
    gradient_accumulation_fusion: bool = Field(
        default=False,
        description="Fuse gradient accumulation to weight gradient computation of linear layers.",
    )
    defer_embedding_wgrad_compute: bool = Field(
        default=False,
        description="Defer vocabulary projection linear layer weight gradient compute to pipeline flush.",
    )
    wgrad_deferral_limit: int = Field(
        default=0,
        description="Number of micro-batches for which weight gradient of vocab projection is deferred.",
    )

    @property
    def model_type(self):
        prefix = self.model_size.split('-')[0]
        return prefix.rstrip('0123456789.')

class GalvatronProfileArgs(BaseModel):
    """Profiling and debugging."""

    profile: int = Field(default=0, description="Whether to profile model GPU memory.")
    profile_mode: Literal["static", "batch", "sequence"] = Field(
        default="static",
        description="Galvatron profiling mode.",
    )
    profile_unit: Literal["attention", "mlp", "all"] = Field(default="all", description="Profile granularity.")
    profile_forward: Literal[0, 1] = Field(default=0, description="Profile forward computation.")
    save_profiled_memory: int = Field(default=0, description="Whether to save profiled memory.")
    exit_after_profiling: Literal[0, 1] = Field(
        default=1,
        description="Whether to exit after profiling time and memory.",
    )


class CommonTrainArgs(BaseModel):
    """Common training args (train_dist.sh TRAIN_ARGS)."""

    seed: Optional[int] = Field(default=42, description="Random seed.")
    iteration: Optional[int] = Field(default=0, ge=0, description="Iteration number.")
    train_iters: Optional[int] = Field(default=None, description="Total number of iterations to train.")
    train_samples: Optional[int] = Field(default=None, description="Total number of samples to train.")
    consumed_train_samples: Optional[int] = Field(default=0, description="Number of samples consumed.")
    eval_iters: Optional[int] = Field(default=1, description="Number of iterations to run for evaluation.")
    eval_interval: Optional[int] = Field(default=1000, description="Number of iterations between evaluations.")
    consumed_valid_samples: Optional[int] = Field(default=0, description="Number of samples consumed for validation.")
    
    skip_train: bool = Field(default=False, description="Whether to skip training.")
    do_train: bool = Field(default=False, description="Whether to do training.")
    do_valid: bool = Field(default=False, description="Whether to do validation.")
    do_test: bool = Field(default=False, description="Whether to do testing.")
    dataloader_type: Literal["single", "cyclic", "external"] = Field(default="single", description="Dataloader type.")
    num_workers: int = Field(default=2, description="Number of workers for dataloader.")
    data_sharding: bool = Field(default=False, description="Whether to shard data across data-parallel ranks in cyclic dataloader.")
    
    lr: Optional[float] = Field(default=None, description="Initial learning rate.")
    min_lr: Optional[float] = Field(default=None, description="Minimum value for learning rate.")
    lr_decay_style: Literal["constant", "linear", "cosine", "inverse-square-root", "WSD"] = Field(
        default="cosine",
        description="Learning rate decay function.",
    )
    lr_warmup_fraction: Optional[float] = Field(default=None, description="Fraction of lr warmup to use.")
    lr_warmup_iters: Optional[int] = Field(default=0, description="Number of warmup iterations (used when lr_warmup_fraction is None).")
    lr_warmup_samples: Optional[int] = Field(default=0, description="Number of warmup samples (used when lr_warmup_fraction is None).")
    lr_warmup_init: float = Field(default=0.0, description="Initial learning rate during warmup.")
    lr_decay_iters: Optional[int] = Field(default=None, description="Number of iterations to decay learning rate.")
    lr_decay_samples: Optional[int] = Field(default=None, description="Number of samples to decay learning rate.")
    lr_wsd_decay_style: Literal["exponential", "linear", "cosine"] = Field(
        default="exponential",
        description="Learning rate decay function for WSD.",
    )
    lr_wsd_decay_iters: Optional[int] = Field(default=None, description="Number of iterations to decay learning rate for WSD.")
    lr_wsd_decay_samples: Optional[int] = Field(default=None, description="Number of samples to decay learning rate for WSD.")
    weight_decay: float = Field(default=0.01, description="Weight decay coefficient for L2 regularization.")
    start_weight_decay: Optional[float] = Field(default=None, description="Initial weight decay coefficient for L2 regularization.")
    end_weight_decay: Optional[float] = Field(default=None, description="End of run weight decay coefficient for L2 regularization.")
    weight_decay_incr_style: Literal["constant", "linear", "cosine"] = Field(
        default="constant",
        description="Weight decay increment function.",
    )
    adam_beta1: float = Field(default=0.9, description="First coefficient for Adam running averages of gradient.")
    adam_beta2: float = Field(default=0.999, description="Second coefficient for Adam running averages of gradient.")
    adam_eps: float = Field(default=1e-8, description="Term added to denominator for numerical stability.")
    init_method_std: float = Field(default=0.02, description="Standard deviation of zero-mean normal for weight init.")

    use_checkpoint_opt_param_scheduler: bool = Field(default=False, description="Whether to use checkpoint values for optimizer param scheduler.")
    override_opt_param_scheduler: bool = Field(default=False, description="Whether to override optimizer param scheduler values with class values.")

    sequence_parallel: bool = Field(default=True, description="Whether to use sequence parallel.")
    global_memory_buffer: bool = Field(default=True, description="Whether to use global memory buffer.")
    use_flash_attn: bool = Field(default=True, description="Use FlashAttention implementation of attention.")

    global_batch_size: Optional[int] = Field(default=None, ge=1, description="Global training batch size.")
    micro_batch_size: Optional[int] = Field(default=None, description="Micro batch size.")
    chunks: int = Field(default=-1, description="Pipeline chunk num.")
    rampup_batch_size: Optional[List[int]] = Field(default=None, description="Rampup batch size. Format: [start_bs, increment, ramp_samples].")
    seq_length: Optional[int] = Field(default=None, description="Maximum sequence length to process.")
    clip_grad: float = Field(default=1.0, ge=0.0, description="Max gradient norm for clipping (0 disables).")

    flash_decode: bool = Field(default=True, description="Use FlashDecode implementation of attention.")
    test_mode: bool = Field(default=False, description="Whether to run real-time tests.")

def _str_to_list(v):
    """Like nargs='*': single str -> [str], list unchanged, None -> None."""
    if v is None:
        return None
    if isinstance(v, str):
        return [v]
    return list(v)


class CommonDataArgs(BaseModel):
    """Common data args (train_dist.sh DATA_ARGS)."""

    data_path: Optional[List[str]] = Field(
        default=None,
        description="Weight-prefix list for train/valid/test datasets split by --split. "
                    "Accepts: (1) a single prefix, (2) weight prefix pairs, (3) a list of prefixes.",
    )
    split: Optional[str] = Field(
        default=None,
        description="Comma-separated proportions for train/valid/test split, e.g. '90,5,5'.",
    )
    train_data_path: Optional[List[str]] = Field(
        default=None,
        description="Weight-prefix list for an independent train dataset.",
    )
    valid_data_path: Optional[List[str]] = Field(
        default=None,
        description="Weight-prefix list for an independent validation dataset.",
    )
    test_data_path: Optional[List[str]] = Field(
        default=None,
        description="Weight-prefix list for an independent test dataset.",
    )

    data_source: Literal["megatron", "hf", "dummy"] = Field(
        default="megatron",
        description="Choice of training data pipeline.",
    )

    # HuggingFace dataset args
    hf_text_keys: Optional[str] = Field(default=None, description="Text keys for HuggingFace dataset.")
    hf_collator_mode: Optional[Literal["packing", "padding"]] = Field(default="padding", description="HuggingFace collator mode.")
    hf_shuffle_buffer_size: Optional[int] = Field(default=0, description="Shuffle buffer size for HuggingFace dataset.")
    hf_num_workers: Optional[int] = Field(default=0, description="Number of workers for HuggingFace dataset.")
    hf_prefetch_factor: Optional[int] = Field(default=2, description="Prefetch factor for HuggingFace dataset.")
    hf_data_mode: Literal["prefetch", "iterable", "mapping"] = Field(
        default="prefetch",
        description="HF pipeline when data_source is hf. "
                    "Accepts: (1) prefetch (multiprocess buffer), (2) iterable (streaming + DataLoader), "
                    "(3) mapping (materialized HFDataset + DataLoader).",
    )

    # Dummy data args
    dummy_sample_mode: Literal["fix_length", "varlen_length"] = Field(default="fix_length", description="Mode for generating dummy samples.")
    dummy_collate_mode: Literal["pack", "padding"] = Field(default="padding", description="Collation mode for dummy data.")
    

    @field_validator("data_path", "train_data_path", "valid_data_path", "test_data_path", mode="before")
    @classmethod
    def str_to_list(cls, v):
        return _str_to_list(v)

    data_args_path: Optional[str] = Field(
        default=None,
        description="Path to a JSON file specifying data-path (useful when the list is too large).",
    )
    per_split_data_args_path: Optional[str] = Field(
        default=None,
        description="Path to a JSON file with 'train', 'valid', 'test' keys for per-split data paths.",
    )
    tokenizer_type: Optional[str] = Field(default="HuggingFaceTokenizer", description="Type of tokenizer to use.")
    tokenizer_model: Optional[str] = Field(default=None, description="SentencePiece tokenizer model path.")
    shared_storage: bool = Field(default=True, description="Cluster is shared storage.")
    num_dataset_builder_threads: int = Field(default=1, description="Number of dataset builder threads.")
    data_cache_path: Optional[str] = Field(default=None, description="Path to cache dataset indices.")
    mmap_bin_files: bool = Field(default=True, description="Whether to mmap the .bin files.")
    s3_cache_path: Optional[str] = Field(default=None, description="Path to cache dataset indices for s3 dataloading.")
    reset_position_ids: bool = Field(default=False, description="Whether to reset position ids after end-of-document token.")
    reset_attention_mask: bool = Field(default=False, description="Whether to reset attention mask after end-of-document token.")
    eod_mask_loss: bool = Field(default=False, description="Whether to mask loss for end-of-document tokens.")
    create_attention_mask_in_dataloader: bool = Field(default=False, description="Whether to create attention mask in dataloader.")
    use_random_dataset: bool = Field(default=False, description="Use random synthetic data instead of real dataset for profiling.")


class CommonCkptArgs(BaseModel):
    """Common checkpoint args (train_dist.sh CKPT_ARGS)."""

    load: Optional[str] = Field(default=None, description="Directory containing a model checkpoint.")
    load_iteration: int = Field(default=0, ge=0, description="Load iteration number.")
    distributed_checkpoint: bool = Field(default=False, description="Whether to use distributed checkpoint.")
    
    save: Optional[str] = Field(default=None, description="Output directory to save checkpoints to.")
    save_interval: Optional[int] = Field(default=None, description="Number of iterations between checkpoint saves.")


# TODO: Add logging code.
class LoggingConfig(BaseModel):
    """Logging config."""

    tensorboard_dir: Optional[str] = Field(default=None, description="Path to save the tensorboard logs.")
    tensorboard_queue_size: int = Field(default=1000, ge=1, description="Size of the tensorboard queue for pending events and summaries before one of the ‘add’ calls forces a flush to disk.")
    wandb_project: str = Field(default='', description="The wandb project name. Ignore wandb by default.")
    wandb_exp_name: str = Field(default='', description="The wandb experiment name.")
    wandb_save_dir: str = Field(default='', description="Path to save the wandb results locally.")

class GalvatronRuntimeArgs(BaseModel):
    """
    Single nested model for all Galvatron runtime/training arguments.
    Covers parallel, model, profile, train, data, ckpt (e.g. train_dist.sh).
    """

    parallel: GalvatronParallelArgs = Field(
        default_factory=GalvatronParallelArgs,
        description="Parallelism and strategy.",
    )
    model: GalvatronModelArgs = Field(
        default_factory=GalvatronModelArgs,
        description="Model and training basics.",
    )
    profile: GalvatronProfileArgs = Field(
        default_factory=GalvatronProfileArgs,
        description="Profiling and debugging.",
    )
    train: CommonTrainArgs = Field(
        default_factory=CommonTrainArgs,
        description="Common training (LR, optimizer, eval).",
    )
    data: CommonDataArgs = Field(
        default_factory=CommonDataArgs,
        description="Common data and tokenizer.",
    )
    ckpt: CommonCkptArgs = Field(
        default_factory=CommonCkptArgs,
        description="Common checkpoint load/save.",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging config.",
    )
    rank: int = Field(default=0, ge=0, description="Rank.")
    world_size: int = Field(default=1, ge=1, description="World size.")
    local_rank: int = Field(default=0, ge=0, description="Local rank.")
    distributed_backend: str = Field(default='nccl', description="Distributed backend.")
    distributed_timeout_minutes: int = Field(default=10, ge=1, description="Distributed timeout minutes.")


# Backward alias: core.args_schema and docs use GalvatronTrainingArgs
GalvatronTrainingArgs = GalvatronRuntimeArgs
