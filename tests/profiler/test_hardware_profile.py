import os

import pytest

from tests.utils.profiler_utils import initialize_hardware_profile_profiler


@pytest.fixture
def base_profiler(profiler_hardware_configs_dir):
    """Create base profiler instance"""
    profiler = initialize_hardware_profile_profiler(profiler_hardware_configs_dir)
    return profiler


def _count_torchrun_blocks(scripts_dir: str, filename: str) -> int:
    """Each profiling command is a block whose first line starts with `torchrun` (echo lines excluded)."""
    path = os.path.join(scripts_dir, filename)
    with open(path, "r") as f:
        return sum(1 for line in f if line.lstrip().startswith("torchrun"))


@pytest.mark.profiler
@pytest.mark.parametrize(
    "num_nodes,num_gpus_per_node,expected_ar,expected_p2p,expected_ar_sp,expected_a2a_sp",
    [
        (1, 4, 3, 2, 22, 22),
        (1, 8, 5, 3, 33, 33),
        (2, 8, 7, 3, 33, 33),
    ],
)
def test_torch_hardware_profile(
    base_profiler,
    num_nodes,
    num_gpus_per_node,
    expected_ar,
    expected_p2p,
    expected_ar_sp,
    expected_a2a_sp,
):
    """Generated scripts use torchrun and profile_*.py (no torch.distributed.launch)."""
    base_profiler.args.num_nodes = num_nodes
    base_profiler.args.num_gpus_per_node = num_gpus_per_node

    path = base_profiler.path
    scripts_dir = os.path.join(path, "scripts")

    base_profiler.profile_bandwidth()
    assert _count_torchrun_blocks(scripts_dir, "profile_allreduce.sh") == expected_ar
    assert _count_torchrun_blocks(scripts_dir, "profile_p2p.sh") == expected_p2p

    base_profiler.profile_sp_bandwidth()
    assert _count_torchrun_blocks(scripts_dir, "profile_allreduce_sp.sh") == expected_ar_sp
    assert _count_torchrun_blocks(scripts_dir, "profile_all2all_sp.sh") == expected_a2a_sp
