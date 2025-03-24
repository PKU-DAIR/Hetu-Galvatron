from galvatron.core import ModelProfiler, initialize_galvatron
from galvatron.models.vit_hf.arguments import model_args, layernum_arg_names, seqlen_arg_names
from galvatron.models.vit_hf.meta_configs import model_name
from galvatron.models.vit_hf.ViTModel_hybrid_parallel import get_vit_config
import os

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='profile')
    
    config = get_vit_config(args, overwrite_args=False)
    
    profiler = ModelProfiler(args)
    
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))
    
    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()