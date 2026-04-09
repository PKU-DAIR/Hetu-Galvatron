from galvatron.core.profiler import HardwareProfiler, ModelProfiler, RuntimeProfiler
from galvatron.core.profiler.args_schema import GalvatronModelProfilerArgs, ProfilerHardwareArgs
from galvatron.core.runtime.args_schema import GalvatronModelArgs
from tests.models.configs.get_config_json import ConfigFactory
from tests.utils.model_utils import ModelFactory


def initialize_model_profile_profiler(profiler_model_configs_dir, model_type, backend, **kwargs):
    """Build a ModelProfiler with Pydantic args matching production (Hydra / args_schema)."""
    _ = (model_type, backend)  # fixture API compatibility
    defaults = dict(
        profile_type="memory",
        profile_mode="static",
        profile_unit="all",
        profile_flow_control="all",
        profile_mixed_precision="bf16",
        profile_fixed_batch_size=8,
        profile_fixed_seq_length_list=[4096],
        profile_layernum_min=1,
        profile_layernum_max=2,
        profile_batch_size_step=1,
        profile_seq_length_step=128,
        profile_max_tp_deg=8,
        runtime_yaml_template_path="scripts/train_dist.yaml",
        model_info=GalvatronModelArgs(model_size="test_model"),
    )
    defaults.update(kwargs)
    args = GalvatronModelProfilerArgs(**defaults)
    profiler = ModelProfiler(args)
    profiler.set_profiler_launcher(str(profiler_model_configs_dir.parent), model_name="test")
    return profiler

def initialize_hardware_profile_profiler(profiler_hardware_configs_dir):
    """Initialize profiler"""

    # Setup search engine
    args = ProfilerHardwareArgs()
    profiler = HardwareProfiler(args)
    profiler.set_path(profiler_hardware_configs_dir)
    return profiler

def initialize_runtime_profile_profiler(profiler_model_configs_dir, model_type, backend, **kwargs):
    """Initialize profiler"""

    # Setup search engine
    class DummyArgs:
        def __init__(self):
            self.profile = True
            self.mixed_precision = 'bf16'
            self.set_model_config_manually = False
            self.set_layernum_manually = False
            self.set_seqlen_manually = False
    args = DummyArgs()
    model_layer_configs, model_name = ModelFactory.get_meta_configs(model_type, backend)
    config_json = ConfigFactory.get_config_json(model_type)
    args.model_size = config_json
    config = ModelFactory.create_config(model_type, backend, args, False)
    # Initialize profiler
    profiler = RuntimeProfiler(args)
    profiler.set_profiler_dist(profiler_model_configs_dir.parent, model_layer_configs(config), model_type, rank = 0, profile_ranks = [0], **kwargs)
    
    return profiler