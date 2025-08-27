def galvatron_training_args(parser, use_megatron=True):
    group = parser.add_argument_group(title="Galvatron Training Arguments")

    group.add_argument(
        "--set_model_config_manually",
        type=int,
        default=0,
        help="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten.",
    )
    group.add_argument(
        "--set_layernum_manually",
        type=int,
        default=0,
        help="Whether to set layernum config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--set_seqlen_manually",
        type=int,
        default=0,
        help="Whether to set sequence length config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--initialize_on_meta",
        type=int,
        default=0,
        help="Whether to initialize parameters on meta device.",
        choices=[0, 1],
    )
    group.add_argument("--global_train_batch_size", type=int, default=32, help="Global training batch size")
    group.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout rate.")
    group.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    group.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam")
    group.add_argument("--check_loss", type=int, default=0, help="Whether to check model correctness.")
    group.add_argument("--profile", type=int, default=0, help="Whether to profile model GPU memory.")
    group.add_argument("--save_profiled_memory", type=int, default=0, help="Whether to save profiled memory.")
    group.add_argument(
        "--profile_type",
        type=str,
        default="allocated",
        help="Profile allocated memory or reserved memory.",
        choices=["allocated", "reserved"],
    )
    group.add_argument(
        "--profile_mode",
        type=str,
        default="static",
        help="Galvatron profiling mode",
        choices=["static", "batch", "sequence"],
    )
    group.add_argument("--load_params", type=int, default=0, help="Whether to load saved init params.")
    group.add_argument(
        "--pp_deg",
        type=int,
        default=2,
        help="Pipeline parallel degree.",
        choices=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    group.add_argument(
        "--global_tp_deg",
        type=int,
        default=-1,
        help="Global tensor parallel degree.",
        choices=[-1, 1, 2, 4, 8, 16, 32],
    )
    group.add_argument(
        "--chunks",
        type=int,
        default=-1,
        help="Pipeline chunk num.",
    )
    group.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    group.add_argument(
        "--sdp",
        type=int,
        default=0,
        help="Apply SDP (zero-3)",
        choices=[0, 1],
    )
    group.add_argument(
        "--galvatron_config_path",
        type=str,
        default=None,
        help="Galvatron strategy config path. If not None, galvatron will run according to json config file.",
    )
    group.add_argument("--global_checkpoint", type=int, default=0, help="Global checkpoint flag.")
    group.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        help="Mixed precision option.",
        choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument(
        "--pipeline_type",
        type=str,
        default="gpipe",
        help="Galvatron pipeline type",
        choices=["gpipe", "pipedream_flush"],
    )
    group.add_argument(
        "--default_dp_type",
        type=str,
        default="ddp",
        help="Default data parallel type",
        choices=["ddp", "zero2", "zero3"],
    )
    group.add_argument(
        "--embed_sdp",
        type=int,
        default=0,
        help="Apply SDP (zero-3) for Embeddings and cls",
        choices=[0, 1],
    )
    group.add_argument(
        "--profile_forward",
        type=int,
        default=0,
        help="Profile forward computation",
        choices=[0, 1],
    )
    group.add_argument(
        "--allow_tf32",
        type=int,
        default=1,
        help="Whether to allow tf32 on Ampere devices",
        choices=[0, 1],
    )
    group.add_argument(
        "--exit_after_profiling",
        type=int,
        default=1,
        help="Whether to exit after profiling time and memory.",
        choices=[0, 1],
    )
    group.add_argument(
        "--shape_order",
        type=str,
        default="SBH",
        help="Model shape order.",
        choices=["SBH", "BSH"],
    )
    group.add_argument(
        "--vocab_tp",
        type=int,
        default=1,
        help="Tensor parallel degree of vocab.",
        choices=[1, 2, 4, 8, 16],
    )
    group.add_argument(
        "--use-ulysses",
        action="store_true",
        help="Whether to use DeepSpeed Ulysses or Megatron-TP",
    )
    group.add_argument(
        "--no_async_grad_reduce",
        action="store_false",
        help="Disable async grad reduce so that gradient will be reduce every micro batch. "
        "Ensure Zero3 memory cost when chunk > 1.",
        dest="async_grad_reduce",
    )
    group.add_argument(
        "--reduce_in_fp32",
        action="store_true",
        help="Use fp32 for gradient reduction.",
    )
    group.add_argument(
        "--entropy_in_fp32",
        action="store_true",
        help="Use fp32 for entropy calculation.",
    )
    group.add_argument(
        "--distributed_checkpoint",
        action="store_true",
        default=False,
        help="Whether to use distributed checkpoint.",
    )
    group.add_argument(
        "--load_iteration",
        type=int,
        default=0,
        help="Load iteration number.",
    )
    if not use_megatron:
        group.add_argument("--lr", type=float, default=1e-4, help="Learning rate of adam")
        group.add_argument("--gpu_id", type=int, default=0, help="Id of GPU to run.")
    else:
        group.add_argument("--no-shared-storage", action="store_false", dest="shared_storage", help="Cluster is not shared storage.")
    
    # MoE arguments
    group.add_argument(
        "--is_moe_model",
        action="store_true",
        help="Whether to use MoE.",
    )
    group.add_argument(
        "--set_experts_manually",
        type=int,
        default=0,
        help="Whether to set experts config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--global_ep_deg",
        type=int,
        default=1,
        help="Experts parallel degree.",
    )
    
    group.add_argument(
        "--global_tp_of_ep_deg",
        type=int,
        default=1,
        help="Tensor parallel degree of experts.",
    )

    group.add_argument(
        "--use_fsep",
        action="store_true",
        help="Whether to use fsep.",
    )

    group.add_argument(
        "--profile_unit",
        choices=["attention", "mlp", "all"],
        default="all",
        help="Profile granularity",
    )

    group.add_argument(
        "--expert_capacity_per_device",
        type=int,
        default=1,
        help="Expert capacity per device.",
    )

    group.add_argument(
        "--recompute_communication",
        action="store_true",
        help="Whether to recompute communication.",
    )
    return parser
