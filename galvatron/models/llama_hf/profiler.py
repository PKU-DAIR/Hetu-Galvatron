import torch
from transformers import LlamaConfig, LlamaForCausalLM
import galvatron
from galvatron.core import GalvatronProfiler, initialize_galvatron
from galvatron.models.llama_hf.arguments import model_args, layernum_arg_names
from galvatron.models.llama_hf.meta_configs import config_from_meta, set_model_config, model_name
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='profile')
    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, overwrite_args=False)
    
    profiler = GalvatronProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
    
    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()