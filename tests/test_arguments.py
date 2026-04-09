import pytest
import argparse

from galvatron.core.profiler.arguments import galvatron_profile_args, galvatron_profile_hardware_args
from galvatron.core.search_engine.arguments import galvatron_search_args


@pytest.mark.utils
def test_galvatron_profile_args():
    """Test galvatron_profile_args function"""
    parser = argparse.ArgumentParser()
    parser = galvatron_profile_args(parser)
    
    # Parse args with default values
    args = parser.parse_args([])
    
    # Test default values
    assert args.profile_type == "memory"
    assert args.set_layernum_manually == 1
    assert args.profile_mode == "static"
    assert args.profile_batch_size_step == 1
    assert args.profile_seq_length_step == 128
    assert args.layernum_min == 1
    assert args.layernum_max == 2
    assert args.max_tp_deg == 8
    assert args.profile_dp_type == "zero3"
    assert args.mixed_precision == "bf16"
    assert not args.use_flash_attn
    assert args.shape_order == "SBH"


@pytest.mark.utils
def test_galvatron_profile_hardware_args():
    """Test galvatron_profile_hardware_args function"""
    parser = argparse.ArgumentParser()
    parser = galvatron_profile_hardware_args(parser)
    
    # Parse args with default values
    args = parser.parse_args([])
    
    # Test default values
    assert args.num_nodes == 1
    assert args.num_gpus_per_node == 8
    assert args.master_addr == "$MASTER_ADDR"
    assert args.master_port == "$MASTER_PORT"
    assert args.node_rank == "$RANK"
    assert args.max_tp_size == 8
    assert args.envs == []
    assert args.max_pp_deg == 8
    assert args.overlap_time_multiply == 4


@pytest.mark.utils
def test_galvatron_search_args():
    """Test galvatron_search_args function"""
    parser = argparse.ArgumentParser()
    parser = galvatron_search_args(parser)
    
    # Parse args with default values
    args = parser.parse_args([])
    
    # Test default values
    assert args.num_nodes == 1
    assert args.num_gpus_per_node == 8
    assert args.memory_constraint == 24
    assert args.min_bsz == 8
    assert args.max_bsz == 10240
    assert args.bsz_scale == 8
    assert args.search_space == "full"
    assert args.sp_space == "tp"
    assert args.max_tp_deg == 8
    assert args.max_pp_deg == 8
    assert args.default_dp_type == "ddp"
    assert args.mixed_precision == "bf16"
    assert args.pipeline_type == "gpipe"
    assert args.use_pipeline_costmodel == 1
    assert args.costmodel_coe == 1.0
    assert args.fine_grained_mode == 1


@pytest.mark.utils
@pytest.mark.parametrize(
    "args_func",
    [
        galvatron_profile_args,
        galvatron_profile_hardware_args,
        galvatron_search_args,
    ],
)
def test_argument_groups(args_func):
    """Test if argument groups are correctly created"""
    parser = argparse.ArgumentParser()
    parser = args_func(parser)
    
    # Check if argument group exists
    group_titles = [group.title for group in parser._action_groups]

    expected_titles = {
        galvatron_profile_args: "Galvatron Profiling Arguments",
        galvatron_profile_hardware_args: "Galvatron Profiling Hardware Arguments",
        galvatron_search_args: "Galvatron Searching Arguments",
    }
    assert expected_titles[args_func] in group_titles
