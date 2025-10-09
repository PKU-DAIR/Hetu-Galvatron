import os
import sys
import types
import torch
for p in ['site_package', 'build/lib']:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), p))

import torch_npu
from mindspeed import megatron_adaptor
from torch_npu.contrib import transfer_to_npu

from functools import wraps
from mindspeed.arguments import process_args

def galvatron_extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            if isinstance(extra_args_provider, list):
                for extra_args in extra_args_provider:
                    parser = extra_args(parser)
            else:
                parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper

def galvatron_parse_args_decorator(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = galvatron_extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper

import megatron
megatron.training.initialize.parse_args = galvatron_parse_args_decorator(megatron.training.initialize.parse_args)
megatron.training.arguments.parse_args = galvatron_parse_args_decorator(megatron.training.arguments.parse_args)
    
# # sys.modules['transformer_engine'] = types.ModuleType('transformer_engine')
# # sys.modules['transformer_engine'].__spec__ = 'te'
# # setattr(sys.modules['transformer_engine'], 'pytorch', torch.nn)
# # setattr(sys.modules['transformer_engine'].pytorch, 'LayerNormLinear', torch.nn.Module)
# # setattr(sys.modules['transformer_engine'].pytorch, 'DotProductAttention', torch.nn.Module)
