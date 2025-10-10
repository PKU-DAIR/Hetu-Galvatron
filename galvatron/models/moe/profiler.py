import os
from galvatron.core import ModelProfiler, initialize_galvatron
from galvatron.models.moe.arguments import layernum_arg_names, model_args
from galvatron.models.moe.MoEModel_hybrid_parallel import get_moe_config
from galvatron.models.moe.meta_configs import model_name

if __name__ == "__main__":
    args = initialize_galvatron(model_args, mode="profile")
    config = get_moe_config(args, overwrite_args=False)
    
    profiler = ModelProfiler(args)
    path = os.path.dirname(os.path.abspath(__file__))
    profiler.set_profiler_launcher(path, layernum_arg_names(), model_name(config))

    if args.profile_unit == "all":
        raise ValueError("Profiling all layers is not supported for MoE models.")
    elif args.profile_unit == "mlp" and args.profile_type == "memory":
        raise NotImplementedError("Memory profiling for MLP layer is not supported now.")
    
    profiler.launch_profiling_scripts()
    profiler.process_profiled_data()
