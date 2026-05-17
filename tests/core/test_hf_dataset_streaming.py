"""Unit tests for ``build_hf_streaming_dataset`` sharding/interleave logic."""
import json
from pathlib import Path

import pytest

pytest.importorskip("datasets")

from galvatron.core.runtime.datasets.huggingface.dataset import (
    FIXED_DATA_SHARD_COUNT,
    build_hf_streaming_dataset,
)


def _write_jsonl(path: Path, n: int):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"text": f"line {i}"}) + "\n")


@pytest.fixture
def tiny_jsonl(tmp_path: Path):
    """A 32-line jsonl — evenly divisible by FIXED_DATA_SHARD_COUNT (=8)."""
    path = tmp_path / "tiny.jsonl"
    _write_jsonl(path, 32)
    return path


@pytest.fixture
def multi_file_json_dir(tmp_path: Path):
    """Two jsonl files so ``build_hf_streaming_dataset`` uses the multi-file
    interleave path (not the single-file ``shard(dp_world_size)`` shortcut)."""
    d = tmp_path / "multi"
    d.mkdir()
    _write_jsonl(d / "a.jsonl", 16)
    _write_jsonl(d / "b.jsonl", 16)
    return d


@pytest.mark.utils
def test_returns_raw_stream_when_dp_not_set(tiny_jsonl):
    ds = build_hf_streaming_dataset(
        train_ds_path=str(tiny_jsonl),
        split="train",
        dp_rank=None,
        dp_world_size=None,
    )
    texts = [ex["text"] for ex in ds]
    assert texts == [f"line {i}" for i in range(32)]


@pytest.mark.utils
def test_dp_world_size_exceeds_shard_count_raises(multi_file_json_dir):
    """Only the multi-file + interleave branch enforces ``dp_world_size <= 8``."""
    with pytest.raises(ValueError, match="FIXED_DATA_SHARD_COUNT"):
        build_hf_streaming_dataset(
            train_ds_path=str(multi_file_json_dir),
            split="train",
            dp_rank=0,
            dp_world_size=FIXED_DATA_SHARD_COUNT + 1,
        )


@pytest.mark.utils
@pytest.mark.parametrize("dp_world_size", [1, 2, 4])
def test_sharded_union_covers_full_dataset(tiny_jsonl, dp_world_size):
    """Across all DP ranks, every original line must be visible at least once."""
    seen = set()
    for rank in range(dp_world_size):
        ds = build_hf_streaming_dataset(
            train_ds_path=str(tiny_jsonl),
            split="train",
            dp_rank=rank,
            dp_world_size=dp_world_size,
        )
        for ex in ds:
            seen.add(ex["text"])
    assert seen == {f"line {i}" for i in range(32)}


@pytest.mark.utils
def test_single_file_round_robin_partition_for_dp2(tiny_jsonl):
    """The single-file fallback should match non-contiguous shard semantics."""
    rank0 = build_hf_streaming_dataset(
        train_ds_path=str(tiny_jsonl),
        split="train",
        dp_rank=0,
        dp_world_size=2,
    )
    rank1 = build_hf_streaming_dataset(
        train_ds_path=str(tiny_jsonl),
        split="train",
        dp_rank=1,
        dp_world_size=2,
    )

    rank0_texts = [ex["text"] for ex in rank0]
    rank1_texts = [ex["text"] for ex in rank1]

    assert rank0_texts == [f"line {i}" for i in range(0, 32, 2)]
    assert rank1_texts == [f"line {i}" for i in range(1, 32, 2)]


@pytest.mark.utils
def test_shuffle_buffer_preserves_content(tiny_jsonl):
    """shuffle_buffer_size>0 only reorders — no records lost, none duplicated."""
    ds = build_hf_streaming_dataset(
        train_ds_path=str(tiny_jsonl),
        split="train",
        shuffle_buffer_size=4,
        seed=0,
    )
    texts = [ex["text"] for ex in ds]
    assert set(texts) == {f"line {i}" for i in range(32)}
    assert len(texts) == 32
