import json
import os

import pytest
from unittest.mock import patch

from tests.utils.profiler_utils import initialize_model_profile_profiler
from tests.utils.profiler_configs import save_profiler_configs
from tests.utils.search_configs import (
    create_static_time_config,
    create_batch_time_config,
    create_sequence_time_config,
    create_static_memory_config,
    create_static_memory_config_sp,
    create_sequence_memory_config_sp,
)


def _reset_profiler_caches(profiler):
    profiler.global_batch_size_list = None
    profiler.layernum_tuple_list = None
    profiler.seq_length_tuple_list = None
    profiler.basic_overrides_dict = None


@pytest.fixture
def base_profiler(profiler_model_configs_dir):
    """Create base profiler instance"""
    profiler = initialize_model_profile_profiler(profiler_model_configs_dir, "llama_search", "hf")
    return profiler


@pytest.mark.profiler
@pytest.mark.parametrize("mode,expected_seq_list,config", [
    ("static", [4096], {"profile_fixed_seq_length_list": [4096]}),
    ("sequence", [128, 256, 384, 512], {
        "profile_min_seq_length": 128,
        "profile_max_seq_length": 512,
        "profile_seq_length_step": 128
    }),
])
def test_get_seq_list(base_profiler, mode, expected_seq_list, config):
    """Test sequence list generation in different modes"""
    base_profiler.args = base_profiler.args.model_copy(update={"profile_mode": mode, "profile_type": "computation", **config})
    _reset_profiler_caches(base_profiler)
    tuples = base_profiler.get_seq_length_tuple_list()
    flat = [t[0] for t in tuples]
    assert flat == expected_seq_list


@pytest.mark.profiler
@pytest.mark.parametrize("mode,expected_bsz_list,config", [
    ("static", [32], {"profile_fixed_batch_size": 32}),
    ("batch", [16, 32, 48, 64], {
        "profile_min_batch_size": 16,
        "profile_max_batch_size": 64,
        "profile_batch_size_step": 16
    }),
])
def test_get_bsz_list(base_profiler, mode, expected_bsz_list, config):
    """Test batch size list generation in different modes"""
    base_profiler.args = base_profiler.args.model_copy(update={"profile_mode": mode, **config})
    _reset_profiler_caches(base_profiler)
    assert base_profiler.get_global_batch_size_list() == expected_bsz_list


@pytest.mark.profiler
@pytest.mark.parametrize("profile_type,profile_mode,expected_calls", [
    # Memory profiling with static mode
    ("memory", "static", {
        "cmd_count": 24,  # Expected number of os.system calls
    }),
    # Memory profiling with sequence mode
    ("memory", "sequence", {
        "cmd_count": 18,  # Reduced because max_tp_deg=1 in sequence mode, sequence length is 128, 256, 512 (different with computation mode)
    }),
    # Computation profiling
    ("computation", "static", {
        "cmd_count": 2,  # 2 layernum_lists * 2 batch_sizes
    }),
    ("computation", "batch", {
        "cmd_count": 4,  # 2 layernum_lists * 2 batch_sizes
    }),
    ("computation", "sequence", {
        "cmd_count": 8,  # 2 layernum_lists * 4 seq_lengths
    })
    
])
def test_launch_profiling_scripts(base_profiler, profile_type, profile_mode, expected_calls):
    """Test launch_profiling_scripts with different configurations"""
    updates = {
        "profile_type": profile_type,
        "profile_mode": profile_mode,
    }
    if profile_type == "computation":
        if profile_mode == "static":
            updates["profile_fixed_batch_size"] = 32
        elif profile_mode == "batch":
            updates["profile_min_batch_size"] = 16
            updates["profile_max_batch_size"] = 32
            updates["profile_batch_size_step"] = 16
        elif profile_mode == "sequence":
            updates["profile_fixed_batch_size"] = 8
            updates["profile_min_seq_length"] = 128
            updates["profile_max_seq_length"] = 512
            updates["profile_seq_length_step"] = 128
    elif profile_mode == "sequence":
        updates["profile_min_seq_length"] = 128
        updates["profile_max_seq_length"] = 512
        updates["profile_seq_length_step"] = 128

    base_profiler.args = base_profiler.args.model_copy(update=updates)
    _reset_profiler_caches(base_profiler)

    env = {
        "NUM_NODES": "1",
        "NUM_GPUS_PER_NODE": "8",
        "RUNTIME_LAUNCHER": "echo",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("os.system") as mock_system:
            base_profiler.launch_profiling_scripts()
            assert mock_system.call_count == expected_calls["cmd_count"]


@pytest.mark.profiler
@pytest.mark.parametrize("mode,config", [
    ("static", {"profile_fixed_batch_size": 8, "profile_layernum_min": 2, "profile_layernum_max": 4}),
    ("batch", {"profile_min_batch_size": 1, "profile_max_batch_size": 10, "profile_batch_size_step": 1, "profile_layernum_min": 2, "profile_layernum_max": 4,}),
    ("sequence", {"profile_fixed_batch_size": 1, "profile_min_seq_length": 4096, "profile_max_seq_length": 32768, "profile_seq_length_step": 4096, "profile_layernum_min": 1, "profile_layernum_max": 2,})
])
def test_process_computation_profiled_data(base_profiler, profiler_model_configs_dir, mode, config):
    """Test processing of computation profiled data"""
    base_profiler.args = base_profiler.args.model_copy(update={"profile_mixed_precision": "bf16", "profile_mode": mode, "profile_type": "computation", **config})
    _reset_profiler_caches(base_profiler)
    save_profiler_configs(
        profiler_model_configs_dir,
        type="computation",
        mode=mode,
        mixed_precision=base_profiler.args.profile_mixed_precision,
        model_name=base_profiler.model_name,
        profile_unit=base_profiler.args.profile_unit,
    )

    base_profiler.process_profiled_data()

    pu = base_profiler.args.profile_unit
    config_path = profiler_model_configs_dir / f"computation_profiling_{base_profiler.args.profile_mixed_precision}_{base_profiler.model_name}_{pu}.json"
    assert config_path.exists()

    with open(config_path) as f:
        loaded = json.load(f)

    if mode == "static":
        result = create_static_time_config()
    elif mode == "batch":
        result = create_batch_time_config()
    else:
        result = create_sequence_time_config()

    for key, value in result.items():
        assert abs(loaded[key] - value) < 1e-6


@pytest.mark.profiler
@pytest.mark.parametrize("mode,config", [
    ("static", {"profile_fixed_batch_size": 8, "profile_layernum_min": 1, "profile_layernum_max": 2, "sequence_parallel": False}),
    ("static", {"profile_fixed_batch_size": 8, "profile_layernum_min": 1, "profile_layernum_max": 2, "sequence_parallel": True}),
    ("sequence", {"profile_fixed_batch_size": 8, "profile_min_seq_length": 512, "profile_max_seq_length": 8192, "profile_layernum_min": 1, "profile_layernum_max": 2, "sequence_parallel": True}),
])
def test_process_memory_profiled_data(base_profiler, profiler_model_configs_dir, mode, config):
    """Test processing of memory profiled data"""
    sp_mode = config["sequence_parallel"]
    base_profiler.args = base_profiler.args.model_copy(update={"profile_mixed_precision": "bf16", "profile_mode": mode, "profile_type": "memory", **config})
    _reset_profiler_caches(base_profiler)
    save_profiler_configs(
        profiler_model_configs_dir,
        type="memory",
        mode=mode,
        mixed_precision=base_profiler.args.profile_mixed_precision,
        model_name=base_profiler.model_name,
        sp_mode=sp_mode,
        profile_unit=base_profiler.args.profile_unit,
    )

    base_profiler.process_profiled_data()

    pu = base_profiler.args.profile_unit
    config_path = profiler_model_configs_dir / f"memory_profiling_{base_profiler.args.profile_mixed_precision}_{base_profiler.model_name}_{pu}.json"
    assert config_path.exists()

    with open(config_path) as f:
        calc_config = json.load(f)

    if mode == "static" and not sp_mode:
        result = create_static_memory_config()
    elif mode == "static" and sp_mode:
        result = create_static_memory_config_sp()
    else:
        result = create_sequence_memory_config_sp()

    def cmp(a, b):
        if isinstance(b, dict):
            for key, value in b.items():
                cmp(a[key], value)
        else:
            assert abs(a - b) < 1e-6

    cmp(calc_config, result)
