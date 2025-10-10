def galvatron_cost_model_args(parser):
    group = parser.add_argument_group(title="Galvatron Searching Arguments")
    
    group.add_argument(
        "--set_model_config_manually", type=int, default=0, help="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten."
    )
    group.add_argument(
        "--set_layernum_manually", type=int, default=0, help="Whether to set layernum config manually (doesn't overwrite other model configs)."
    )
    group.add_argument(
        "--set_seqlen_manually", type=int, default=0, help="Whether to set sequence length config manually (doesn't overwrite other model configs)."
    )
    group.add_argument(
        "--set_experts_manually",
        type=int,
        default=0,
        help="Whether to set experts config manually (doesn't overwrite other model configs).",
    )
    group.add_argument(
        "--num_nodes", type=int, default=1, help="Number of Nodes.",
    )
    group.add_argument(
        "--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node.",
    )
    group.add_argument(
        "--memory_constraint", type=int, default=24, help="Memory constraint of Galvatron",
    )
    group.add_argument(
        "--search_space", type=str, default="full", help="Galvatron parallelism optimization type.", choices=["full","dp+tp","dp+pp", "3d", "dp", "sdp", "tp", "pp"],
    )
    group.add_argument(
        "--sp_space", type=str, default="tp", help="Galvatron sequence parallelism optimization type.", choices=["tp+sp","tp","sp"],
    )
    group.add_argument(
        "--default_dp_type", type=str, default="ddp", help="Default data parallel type", choices=["ddp","zero2"],
    )
    group.add_argument(
        "--disable_vtp", type=int, default=0, help="Whether to disable vocab tp."
    )
    # group.add_argument(
    #     "--embed_sdp", type=int, default=0, help="Apply SDP (zero-3) for Embeddings and cls", choices=[0, 1],
    # )
    group.add_argument(
        "--mixed_precision", type=str, default="bf16", help="Mixed precision option.", choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument(
        "--pipeline_type", type=str, default="gpipe", help="Galvatron pipeline type", choices=["gpipe","pipedream_flush"],
    )
    group.add_argument(
        "--costmodel_coe", type=float, default=1.0, help="Multiply the outcome of time cost model by this coefficient. Only for fine-tuning time cost model, should be 1.0 in default.",
    )
    group.add_argument(
        "--sequence_parallel", action="store_true", help="Whether to use sequence parallel",
    )
    group.add_argument(
        "--no_global_memory_buffer", action="store_false",
        help='Disable the estimation of global memory for all gather buffer when using Megatron-SP.',
        dest='global_memory_buffer'
    )
    group.add_argument(
        "--no_async_grad_reduce", action="store_false",
        help='Disable async grad reduce so that gradient will be reduce every micro batch. '
        'Ensure Zero3 memory cost when chunk > 1.',
        dest='async_grad_reduce'
    )
    group.add_argument(
        "--memory_profiling_path", type=str, default=None, help="Path to memory profiling config."
    )
    group.add_argument(
        "--time_profiling_path", type=str, default=None, help="Path to time profiling config."
    )
    group.add_argument(
        "--allreduce_bandwidth_config_path", type=str, default=None, help="Path to allreduce bandwidth config."
    )
    group.add_argument(
        "--p2p_bandwidth_config_path", type=str, default=None, help="Path to p2p bandwidth config."
    )
    group.add_argument(
        "--overlap_coe_path", type=str, default=None, help="Path to overlap coefficient config."
    )
    group.add_argument(
        "--sp_time_path", type=str, default=None, help="Path to sequence parallelism time config."
    )
    group.add_argument(
        "--output_config_path", type=str, default=None, help="Path to output config."
    )
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    
    group.add_argument("--fine_grained_mode", type=int, default=1, help="Enable fine-grained search.")
    group.add_argument(
        "--time_profile_mode", type=str, default="static", help="Galvatron profiling mode", choices=["static", "batch", "sequence", "hybrid"]
    )
    group.add_argument(
        "--memory_profile_mode", type=str, default="static", help="Galvatron profiling mode", choices=["static", "batch", "sequence", "hybrid"]
    )
    group.add_argument(
        "--parallel_search", action="store_true", help="Enable parallel search for faster execution"
    )
    group.add_argument(
        "--worker", type=int, default=0, help="Number of worker threads for parallel search. Default is 2x CPU cores"
    )
    group.add_argument(
        "--log_dir", type=str, default="logs", help="Log directory for search engine."
    )
    
    group.add_argument(
        "--profile_granularity", type=str, default="together", help="Granularity of profiling.", choices=["together", "split"]
    )   
    
    group.add_argument(
        "--estimate_tp_time_type", type=str, default="fixed", help="Estimate tensor parallel time by fixed value or fitted curve.", choices=["fixed", "fit"]
    )
    group.add_argument(
        "--zero_with_slight_noise", type=int, default=1, help="Whether to add slight noise to zero memory estimation to improve optimization stability."
    )
    return parser
