"""Unit tests for PrefetchBuffer."""
import pytest
import torch

# ``prefetch_strategy`` transitively imports ``dataset.py`` which requires the
# HuggingFace ``datasets`` package. Skip cleanly when it is not available.
pytest.importorskip("datasets")

from galvatron.core.runtime.datasets.huggingface.collator import IGNORE_INDEX
from galvatron.core.runtime.datasets.huggingface.prefetch_strategy import (
    PrefetchBuffer,
)


@pytest.mark.utils
class TestPrefetchBufferPadding:
    def test_alloc_shapes(self):
        buf = PrefetchBuffer(
            num_entries=2, max_batch_size=4, seq_length=8, mode="padding"
        )
        assert len(buf.entries) == 2
        e = buf.entries[0]
        assert e["tokens"].shape == (4, 8)
        assert e["labels"].shape == (4, 8)
        assert e["loss_mask"].shape == (4, 8)
        assert e["position_ids"].shape == (4, 8)
        assert e["valid_lens"].shape == (4,)

    def test_roundtrip_flash(self):
        buf = PrefetchBuffer(
            num_entries=2, max_batch_size=4, seq_length=6,
            mode="padding", use_flash_attn=True,
        )
        batch = {
            "tokens": torch.tensor(
                [[1, 2, 3, 0, 0, 0], [7, 8, 0, 0, 0, 0]], dtype=torch.long
            ),
            "labels": torch.tensor(
                [[2, 3, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
                 [IGNORE_INDEX] * 6],
                dtype=torch.long,
            ),
            "loss_mask": torch.tensor(
                [[1, 1, 0, 0, 0, 0], [0] * 6], dtype=torch.float
            ),
            "position_ids": torch.arange(6).unsqueeze(0).expand(2, 6).contiguous(),
        }
        valid_lens = torch.tensor([3, 1], dtype=torch.long)

        actual_bs = buf.write_padding(0, batch, valid_lens)
        assert actual_bs == 2

        out = buf.read_padding(0, actual_bs)
        assert torch.equal(out["tokens"], batch["tokens"])
        assert torch.equal(out["labels"], batch["labels"])
        assert torch.equal(out["loss_mask"], batch["loss_mask"])
        assert torch.equal(out["position_ids"], batch["position_ids"])
        # flash path: the dense 4D attention mask is NOT reconstructed
        assert "attention_mask" not in out

    def test_roundtrip_rebuild_attention_mask(self):
        buf = PrefetchBuffer(
            num_entries=1, max_batch_size=2, seq_length=4,
            mode="padding", use_flash_attn=False,
        )
        batch = {
            "tokens": torch.tensor([[1, 2, 0, 0]], dtype=torch.long),
            "labels": torch.tensor(
                [[2, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]], dtype=torch.long
            ),
            "loss_mask": torch.tensor([[1, 0, 0, 0]], dtype=torch.float),
            "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        }
        valid_lens = torch.tensor([2], dtype=torch.long)

        buf.write_padding(0, batch, valid_lens)
        out = buf.read_padding(0, 1)

        assert out["attention_mask"].shape == (1, 1, 4, 4)
        am = out["attention_mask"]
        # padded key columns (2, 3) must be masked for every query row
        assert bool(am[0, 0, 0, 2]) is True
        assert bool(am[0, 0, 0, 3]) is True
        # query 0 attending to key 0 should not be masked
        assert bool(am[0, 0, 0, 0]) is False

    def test_write_smaller_than_max_bs(self):
        buf = PrefetchBuffer(
            num_entries=1, max_batch_size=4, seq_length=4,
            mode="padding", use_flash_attn=True,
        )
        batch = {
            "tokens": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4, IGNORE_INDEX]], dtype=torch.long),
            "loss_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.float),
            "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        }
        valid_lens = torch.tensor([3], dtype=torch.long)

        actual_bs = buf.write_padding(0, batch, valid_lens)
        assert actual_bs == 1

        out = buf.read_padding(0, actual_bs)
        assert out["tokens"].shape == (1, 4)
        assert torch.equal(out["tokens"], batch["tokens"])

@pytest.mark.utils
class TestPrefetchBufferPacking:
    def test_alloc_shapes(self):
        buf = PrefetchBuffer(
            num_entries=1, max_batch_size=3, seq_length=4, mode="packing"
        )
        e = buf.entries[0]
        assert e["input_ids"].shape == (1, 3 * 4)
        assert e["labels"].shape == (1, 3 * 4)
        assert e["cu_seqlens"].shape == (3 + 1,)
        assert e["cu_seqlens"].dtype == torch.int32

    def test_roundtrip(self):
        buf = PrefetchBuffer(
            num_entries=1, max_batch_size=4, seq_length=4, mode="packing"
        )
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        labels    = torch.tensor(
            [[2, 3, IGNORE_INDEX, 5, IGNORE_INDEX]], dtype=torch.long
        )
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        batch = (input_ids, {"labels": labels, "cu_seqlens": cu_seqlens}, None)

        packed_len, num_seqs = buf.write_packing(0, batch)
        assert packed_len == 5
        assert num_seqs == 3  # == len(cu_seqlens)

        out_ids, out_kwargs, loss_fn = buf.read_packing(0, packed_len, num_seqs)
        assert torch.equal(out_ids, input_ids)
        assert torch.equal(out_kwargs["labels"], labels)
        assert torch.equal(out_kwargs["cu_seqlens"], cu_seqlens)
        assert out_kwargs["attention_mask"] is None
        assert out_kwargs["rotary_embedding"] is None
        assert callable(loss_fn)
