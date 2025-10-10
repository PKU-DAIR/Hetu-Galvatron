from megatron.training.initialize import initialize_megatron
from megatron.training import get_args as get_megatron_args
import argparse
from .profiler import galvatron_profile_args, galvatron_profile_hardware_args
from .runtime.arguments import galvatron_training_args
from .search_engine.arguments import galvatron_search_args
from .cost_model.arguments import galvatron_cost_model_args

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
    elif mode == "cost_model":
        extra_args_provider = [galvatron_cost_model_args]
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
