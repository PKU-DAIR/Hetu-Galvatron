def galvatron_profile_args(parser):
    group = parser.add_argument_group(title="Galvatron Profiling Arguments")

    group.add_argument(
        "--profile_type", type=str, default="memory", help="Galvatron profiling type", choices=["memory", "computation"]
    )
    group.add_argument(
        "--set_model_config_manually",
        type=int,
        default=0,
        help="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten.",
    )
    group.add_argument(
        "--set_layernum_manually",
        type=int,
        default=1,
        help="Whether to set layernum config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--set_seqlen_manually",
        type=int,
        default=0,
        help="Whether to set sequence length config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--set_experts_manually",
        type=int,
        default=0,
        help="Whether to set experts config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--profile_mode",
        type=str,
        default="static",
        help="Galvatron profiling mode",
        choices=["static", "batch", "sequence"],
    )
    group.add_argument("--profile_batch_size", type=int, default=None, help="Galvatron profiling batch size")
    group.add_argument("--profile_min_batch_size", type=int, default=None, help="Galvatron profiling min batch size")
    group.add_argument("--profile_max_batch_size", type=int, default=None, help="Galvatron profiling max batch size")
    group.add_argument("--profile_batch_size_step", type=int, default=1, help="Galvatron profiling batch size step")
    group.add_argument(
        "--profile_seq_length_list", type=str, default=None, help="Galvatron profiling sequence length step"
    )
    group.add_argument(
        "--profile_min_seq_length", type=int, default=None, help="Galvatron profiling max sequence length"
    )
    group.add_argument(
        "--profile_max_seq_length", type=int, default=None, help="Galvatron profiling max sequence length"
    )
    group.add_argument(
        "--profile_seq_length_step", type=int, default=128, help="Galvatron profiling sequence length step"
    )
    group.add_argument("--layernum_min", type=int, default=1, help="Layernum min for profiling.")
    group.add_argument("--layernum_max", type=int, default=2, help="Layernum min for profiling.")
    group.add_argument("--max_tp_deg", type=int, default=8, help="Maximum tensor parallel degree to profile.")
    group.add_argument(
        "--profile_dp_type", type=str, default="zero3", help="Use zero3 or ddp to profile.", choices=["zero3", "ddp"]
    )
    group.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        help="Mixed precision option.",
        choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument("--use-flash-attn", action="store_true", help="Use FlashAttention implementation of attention.")
    group.add_argument("--extra_args_str", type=str, default="", help="Extra arguments for megatron initilization.")

    group.add_argument(
        "--sequence_parallel",
        action="store_true",
        help="Whether to use sequence parallel",
    )

    group.add_argument(
        "--shape_order",
        type=str,
        default="SBH",
        help="Model shape order.",
        choices=["SBH", "BSH"],
    )

    group.add_argument(
        "--make-vocab-size-divisible-by",
        type=int,
        default=128,
        help="Pad the vocab size to be divisible by this value." "This is added for computational efficieny reasons.",
    )

    group.add_argument(
        "--profile_unit",
        choices=["attention", "mlp", "all"],
        default="all",
        help="Profile granularity",
    )

    return parser


def galvatron_profile_hardware_args(parser):
    group = parser.add_argument_group(title="Galvatron Profiling Hardware Arguments")

    group.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of Nodes.",
    )
    group.add_argument(
        "--num_gpus_per_node",
        type=int,
        default=8,
        help="Number of GPUs per node.",
    )
    group.add_argument(
        "--master_addr",
        type=str,
        default="$MASTER_ADDR",
        help="Master address.",
    )
    group.add_argument(
        "--master_port",
        type=str,
        default="$MASTER_PORT",
        help="Master port.",
    )
    group.add_argument(
        "--node_rank",
        type=str,
        default="$RANK",
        help="Node rank.",
    )
    group.add_argument(
        "--max_tp_size",
        type=int,
        default=8,
        help="Maximum tensor parallel size.",
    )
    group.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=[],
        help="Additional environment variables in format KEY=VALUE",
    )
    group.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="Backend of nccl-tests.",
        choices=["nccl", "torch"],
    )
    group.add_argument(
        "--nccl_test_dir",
        type=str,
        default="nccl-tests",
        help="Directory of nccl-tests.",
    )
    group.add_argument(
        "--mpi_path",
        type=str,
        default="/usr/local/mpi/",
        help="MPI Path.",
    )
    group.add_argument(
        "--start_mb",
        type=int,
        default=16,
        help="Starting communication size in MB.",
    )
    group.add_argument(
        "--end_mb",
        type=int,
        default=512,
        help="Ending communication size in MB.",
    )
    group.add_argument(
        "--scale",
        type=int,
        default=2,
        help="Memory scale of nccl-tests.",
    )
    group.add_argument(
        "--hostfile",
        type=str,
        default="hostfile",
        help="Hostfile for nccl-tests.",
    )
    group.add_argument(
        "--avg_or_min_or_first",
        type=str,
        default="first",
        help="For a given group size, if 'first', only profile first group;"
             "if 'min', profile the group with minimum bandwidth;"
             "if 'avg', profile all groups and take the average bandwidth.",
        choices=["first", "min", "avg"],
    )
    group.add_argument("--max_pp_deg", type=int, default=8, help="Maximum pipeline parallel degree to search.")
    group.add_argument(
        "--overlap_time_multiply",
        type=int,
        default=4,
        help="The multiple of communication time and computation time when overlapped.",
    )

    return parser
