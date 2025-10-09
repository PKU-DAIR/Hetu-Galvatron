from megatron.training.initialize import initialize_megatron
from megatron.training import get_args as get_megatron_args
import argparse

def initialize_galvatron(model_args = None, mode="train_dist"):
    use_megatron = False
    if mode in ["train_dist", "train"]:
        use_megatron = (mode == "train_dist")
        extra_args_provider = [lambda parser: galvatron_training_args(parser, use_megatron)]
    elif mode == "profile":
        extra_args_provider = [galvatron_profile_args]
    elif mode == "search":
        extra_args_provider = [galvatron_search_args]
    elif mode == "profile_hardware":
        extra_args_provider = [galvatron_profile_hardware_args]
    if model_args is not None:
        extra_args_provider.append(model_args)
    if use_megatron:
        initialize_megatron(extra_args_provider)
        args = get_args()
    else:
        args = parse_args(extra_args_provider)
    if 'allow_tf32' in args and args.allow_tf32:
        import torch
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return args

def parse_args(extra_args_provider):
    parser = argparse.ArgumentParser()
    # Custom arguments.
    if extra_args_provider is not None:
        if isinstance(extra_args_provider, list):
            for extra_args in extra_args_provider:
                parser = extra_args(parser)
        else:
            parser = extra_args_provider(parser)
    args = parser.parse_args()
    return args

def get_args():
    return get_megatron_args()

def galvatron_training_args(parser, use_megatron=True):
    group = parser.add_argument_group(title="Galvatron Training Arguments")

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
        "--initialize_on_meta", type=int, default=0, help="Whether to initialize parameters on meta device.", choices=[0, 1]
    )
    group.add_argument(
        "--global_train_batch_size", type=int, default=32, help="Global training batch size"
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    group.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    group.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    group.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    group.add_argument(
        "--save_profiled_memory", type=int, default=0, help="Whether to save profiled memory."
    )
    group.add_argument(
        "--profile_type", type=str, default="allocated", help="Profile allocated memory or reserved memory.",
        choices = ["allocated", "reserved"],
    )
    group.add_argument(
        "--profile_mode", type=str, default="static", help="Galvatron profiling mode", choices=["static", "batch", "sequence"]
    )
    group.add_argument(
        "--load_params", type=int, default=0, help="Whether to load saved init params."
    )
    group.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16,32,64,128,256,512],
    )
    group.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32],
    )
    group.add_argument(
        "--chunks", type=int, default=-1, help="Pipeline chunk num.",
    )
    group.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    group.add_argument(
        "--sdp", type=int, default=0, help="Apply SDP (zero-3)", choices=[0, 1],
    )
    group.add_argument(
        "--galvatron_config_path", type=str, default=None, help="Galvatron strategy config path. If not None, galvatron will run according to json config file.",
    )
    group.add_argument(
        "--global_checkpoint", type=int, default=0, help="Global checkpoint flag."
    )
    group.add_argument(
        "--mixed_precision", type=str, default="bf16", help="Mixed precision option.", choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument(
        "--pipeline_type", type=str, default="gpipe", help="Galvatron pipeline type", choices=["gpipe","pipedream_flush"],
    )
    group.add_argument(
        "--default_dp_type", type=str, default="ddp", help="Default data parallel type", choices=["ddp","zero2","zero3"],
    )
    group.add_argument(
        "--embed_sdp", type=int, default=0, help="Apply SDP (zero-3) for Embeddings and cls", choices=[0, 1],
    )
    group.add_argument(
        "--profile_forward", type=int, default=0, help="Profile forward computation", choices=[0, 1],
    )
    group.add_argument(
        "--allow_tf32", type=int, default=1, help="Whether to allow tf32 on Ampere devices", choices=[0, 1],
    )
    group.add_argument(
        "--exit_after_profiling", type=int, default=1, help="Whether to exit after profiling time and memory.", choices=[0, 1],
    )
    group.add_argument(
        "--shape_order", type=str, default='SBH', help="Model shape order.", choices=['SBH', 'BSH'],
    )
    group.add_argument(
        "--vocab_tp", type=int, default=1, help="Tensor parallel degree of vocab.", choices=[1,2,4,8,16],
    )
    group.add_argument(
        "--use-ulysses", action="store_true", help="Whether to use DeepSpeed Ulysses or Megatron-TP",
    )
    group.add_argument(
        "--no_async_grad_reduce", action="store_false",
        help='Disable async grad reduce so that gradient will be reduce every micro batch. '
        'Ensure Zero3 memory cost when chunk > 1.',
        dest='async_grad_reduce'
    )
    group.add_argument(
        "--reduce_in_fp32", action="store_true", help="Use fp32 for gradient reduction.",
    )
    group.add_argument(
        "--entropy_in_fp32", action="store_true", help="Use fp32 for entropy calculation.",
    )
    if not use_megatron:
        group.add_argument("--lr", type=float, default=1e-4, help="Learning rate of adam")
        group.add_argument("--gpu_id", type=int, default=0, help="Id of GPU to run.")
        group.add_argument("--local-rank", type=int, default=0, help="Local rank.")
    else:
        group.add_argument("--local-rank", type=int, default=-1, help="Local rank.")
    return parser

def galvatron_profile_args(parser):
    group = parser.add_argument_group(title="Galvatron Profiling Arguments")

    group.add_argument(
        "--profile_type", type=str, default="memory", help="Galvatron profiling type", choices=["memory", "computation"]
    )
    group.add_argument(
        "--set_model_config_manually", type=int, default=0, help="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten."
    )
    group.add_argument(
        "--set_layernum_manually", type=int, default=1, help="Whether to set layernum config manually (doesn't overwrite other model configs)."
    )
    group.add_argument(
        "--set_seqlen_manually", type=int, default=0, help="Whether to set sequence length config manually (doesn't overwrite other model configs)."
    )
    group.add_argument(
        "--profile_mode", type=str, default="static", help="Galvatron profiling mode", choices=["static", "batch", "sequence", "hybrid"]
    )
    group.add_argument(
        "--profile_batch_size", type=int, default=None, help="Galvatron profiling batch size"
    )
    group.add_argument(
        "--profile_min_batch_size", type=int, default=None, help="Galvatron profiling min batch size"
    )
    group.add_argument(
        "--profile_max_batch_size", type=int, default=None, help="Galvatron profiling max batch size"
    )
    group.add_argument(
        "--profile_batch_size_step", type=int, default=1, help="Galvatron profiling batch size step"
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
    group.add_argument(
        "--layernum_min", type=int, default=1, help="Layernum min for profiling."
    )
    group.add_argument(
        "--layernum_max", type=int, default=2, help="Layernum min for profiling."
    )
    group.add_argument(
        "--max_tp_deg", type=int, default=8, help="Maximum tensor parallel degree to profile."
    )
    group.add_argument(
        "--profile_dp_type", type=str, default="zero3", help="Use zero3 or ddp to profile.", choices=["zero3","ddp"]
    )
    group.add_argument(
        "--mixed_precision", type=str, default="bf16", help="Mixed precision option.", choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument("--use-fused-rmsnorm", action='store_true',
                       help="Use fused rmsnorm.")
    group.add_argument("--use-fused-swiglu", action='store_true',
                       help="Use fused swiglu.")
    group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                       help="Use fused rotary-pos-emb.")
    
    group.add_argument(
        "--use-flash-attn", action="store_true", help="Use FlashAttention implementation of attention."
    )
    group.add_argument(
        "--extra_args_str", type=str, default="", help="Extra arguments for megatron initilization."
    )
    
    group.add_argument(
        "--sequence_parallel", action="store_true", help="Whether to use sequence parallel",
    )
    
    group.add_argument(
        "--shape_order", type=str, default='SBH', help="Model shape order.", choices=['SBH', 'BSH'],
    )
    
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    
    return parser

def galvatron_profile_hardware_args(parser):
    group = parser.add_argument_group(title="Galvatron Profiling Hardware Arguments")
    
    group.add_argument(
        "--num_nodes", type=int, default=1, help="Number of Nodes.",
    )
    group.add_argument(
        "--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node.",
    )
    group.add_argument(
        "--node_rank", type=int, default= -1, help="Rank of node.",
    )
    group.add_argument(
        "--master_addr", type=str, default= 'localhost', help="Master address.",
    )
    group.add_argument(
        "--master_port", type=int, default= '12345', help="Master port.",
    )
    group.add_argument(
        "--nccl_test_dir", type=str, default='nccl-tests', help="Directory of nccl-tests.",
    )
    group.add_argument(
        "--ascend_dir", type=str, default='/usr/local/Ascend/ascend-toolkit/latest', help="Directory of ascend toolkit.",
    )
    group.add_argument(
        "--mpi_path", type=str, default='/usr/local/mpi/', help="MPI Path.",
    )
    group.add_argument(
        "--start_mb", type=int, default=16, help="Starting communication size in MB.",
    )
    group.add_argument(
        "--end_mb", type=int, default=512, help="Ending communication size in MB.",
    )
    group.add_argument(
        "--scale", type=int, default=2, help="Memory scale of nccl-tests.",
    )
    group.add_argument(
        "--hostfile", type=str, default='hostfile', help="Hostfile for nccl-tests.",
    )
    group.add_argument(
        "--avg_or_min_or_first", type=str, default='first', help="For a given group size, if 'first', only profile first group; if 'min', profile the group with minimum bandwidth; if 'avg', profile all groups and take the average bandwidth.", choices=['first', 'min', 'avg'],
    )
    group.add_argument(
        "--max_pp_deg", type=int, default=8, help="Maximum pipeline parallel degree to search."
    )
    group.add_argument(
        "--overlap_time_multiply", type=int, default=4, help='The multiple of communication time and computation time when overlapped.'
    )
    
    return parser


def galvatron_search_args(parser):
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
        "--num_nodes", type=int, default=1, help="Number of Nodes.",
    )
    group.add_argument(
        "--num_gpus_per_node", type=int, default=8, help="Number of GPUs per node.",
    )
    group.add_argument(
        "--memory_constraint", type=int, default=24, help="Memory constraint of Galvatron",
    )
    group.add_argument(
        "--min_bsz", type=int, default=8, help="Min batch size for searching.",
    )
    group.add_argument(
        "--max_bsz", type=int, default=10240, help="Max batch size for searching.",
    )
    group.add_argument(
        "--recommend_min_bsz", type=int, default=0, help="If 1, start searching from a recommended bsz to accelerate optimization.",
    )
    group.add_argument(
        "--settle_bsz", type=int, default=-1, help="If > 1, only search bsz=settle_bsz."
    )
    group.add_argument(
        "--settle_chunk", type=int, default=-1, help="If > 1, only search chunk=settle_chunk."
    )
    group.add_argument(
        "--bsz_scale", type=int, default=8, help="Bsz scale for searching.",
    )
    group.add_argument(
        "--search_space", type=str, default="full", help="Galvatron parallelism optimization type.", choices=["full","dp+tp","dp+pp", "3d", "dp", "sdp", "tp", "pp"],
    )
    group.add_argument(
        "--sp_space", type=str, default="tp", help="Galvatron sequence parallelism optimization type.", choices=["tp+sp","tp","sp"],
    )
    group.add_argument(
        "--disable_dp", type=int, default=0, help="Whether to disable dp."
    )
    group.add_argument(
        "--disable_tp", type=int, default=0, help="Whether to disable tp."
    )
    group.add_argument(
        "--disable_vtp", type=int, default=0, help="Whether to disable vocab tp."
    )
    group.add_argument(
        "--disable_pp", type=int, default=0, help="Whether to disable pp."
    )
    group.add_argument(
        "--disable_sdp", type=int, default=0, help="Whether to disable sdp."
    )
    group.add_argument(
        "--disable_ckpt", type=int, default=0, help="Whether to disable checkpoint"
    )
    group.add_argument(
        "--disable_tp_consec", type=int, default=0, help="Whether to disable tp_consec."
    )
    group.add_argument(
        "--max_tp_deg", type=int, default=8, help="Maximum tensor parallel degree to search."
    )
    group.add_argument(
        "--max_pp_deg", type=int, default=8, help="Maximum pipeline parallel degree to search."
    )
    group.add_argument(
        "--default_dp_type", type=str, default="ddp", help="Default data parallel type", choices=["ddp","zero2"],
    )
    group.add_argument(
        "--embed_sdp", type=int, default=0, help="Apply SDP (zero-3) for Embeddings and cls", choices=[0, 1],
    )
    group.add_argument(
        "--mixed_precision", type=str, default="bf16", help="Mixed precision option.", choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument(
        "--pipeline_type", type=str, default="gpipe", help="Galvatron pipeline type", choices=["gpipe","pipedream_flush"],
    )
    group.add_argument(
        "--use_pipeline_costmodel", type=int, default=1, help="Whether to use pipeline cost model.",
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
    
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    
    group.add_argument("--fine_grained_mode", type=int, default=1, help="Enable fine-grained search.")
    group.add_argument(
        "--profile_mode", type=str, default="static", help="Galvatron profiling mode", choices=["static", "batch", "sequence", "hybrid"]
    )
    return parser