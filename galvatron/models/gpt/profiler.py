import os
import sys

from galvatron.core.arguments import load_with_hydra
from galvatron.core.profiler.model_profiler import ModelProfiler

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml")):
        config_path, overrides = sys.argv[1], sys.argv[2:]
        sys.argv = [sys.argv[0]]
        args = load_with_hydra(config_path, overrides=overrides, mode="model_profiler")
    else:
        raise ValueError("Usage: python profiler.py <config_path> [overrides...]")

    model_profiler = ModelProfiler(args)

    path = os.path.dirname(os.path.abspath(__file__))
    model_profiler.set_profiler_launcher(
        path=path,
        model_name=args.model_info.model_size,
    )
    model_profiler.launch_profiling_scripts()
    model_profiler.process_profiled_data() 