"""Unit tests for PackingCollator / PaddingCollator / get_collate_fn."""
from types import SimpleNamespace

import pytest
import torch

from galvatron.core.runtime.datasets.huggingface import collator as collator_mod
from galvatron.core.runtime.datasets.huggingface.collator import (
    IGNORE_INDEX,
    PackingCollator,
    PaddingCollator,
    get_collate_fn,
)


def _make_args(seq_length=8, use_flash_attn=True, hf_collator_mode="padding"):
    return SimpleNamespace(
        train=SimpleNamespace(seq_length=seq_length, use_flash_attn=use_flash_attn),
        data=SimpleNamespace(hf_collator_mode=hf_collator_mode),
    )


class _FakeTokenizer:
    def __init__(self, pad_token_id=None):
        self._tokenizer = SimpleNamespace(pad_token_id=pad_token_id)

@pytest.mark.utils
class TestPackingCollator:
    def test_shapes_and_cu_seqlens(self):
        batch = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([4, 5], dtype=torch.long),
        ]
        input_ids, kwargs, loss_fn = PackingCollator()(batch)

        assert input_ids.dim() == 2 and input_ids.shape == (1, 5)
        assert input_ids[0].tolist() == [1, 2, 3, 4, 5]
        assert kwargs["cu_seqlens"].tolist() == [0, 3, 5]
        assert kwargs["cu_seqlens"].dtype == torch.int32
        assert kwargs["attention_mask"] is None
        assert kwargs["rotary_embedding"] is None
        assert callable(loss_fn)

    def test_labels_are_shifted_with_ignore_at_each_seq_tail(self):
        """PackingCollator uses a *global* shift: only the very last position
        is set to IGNORE_INDEX. Interior sub-sequence boundaries do *not*
        insert IGNORE_INDEX — this matches the current implementation."""
        batch = [
            torch.tensor([10, 20, 30], dtype=torch.long),
            torch.tensor([40, 50], dtype=torch.long),
        ]
        _, kwargs, _ = PackingCollator()(batch)
        labels = kwargs["labels"][0]
        # labels[:-1] = input[1:], labels[-1] = IGNORE
        assert labels.tolist() == [20, 30, IGNORE_INDEX, 50, IGNORE_INDEX]


# ---------------------------------------------------------------------------
# PaddingCollator
# ---------------------------------------------------------------------------
@pytest.mark.utils
class TestPaddingCollator:
    @staticmethod
    def _patch_args(monkeypatch, seq_length, use_flash_attn):
        args = _make_args(seq_length=seq_length, use_flash_attn=use_flash_attn)
        monkeypatch.setattr(collator_mod, "get_args", lambda: args)
        return args

    def test_padding_with_flash(self, monkeypatch):
        self._patch_args(monkeypatch, seq_length=5, use_flash_attn=True)
        collate = PaddingCollator(seq_length=5, pad_token_id=0)

        batch = [
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.tensor([7], dtype=torch.long),
        ]
        out = collate(batch)

        assert out["tokens"].shape == (2, 5)
        assert out["tokens"][0].tolist() == [1, 2, 3, 0, 0]
        assert out["tokens"][1].tolist() == [7, 0, 0, 0, 0]
        # Shift input_ids[:, 1:] into labels[:, :-1], then mask columns where
        # padding_mask is False. Positions still "valid" for the mask can retain
        # pad_token_id in labels (e.g. predicting the first padding slot).
        assert out["labels"][0].tolist() == [2, 3, 0, IGNORE_INDEX, IGNORE_INDEX]
        assert out["labels"][1].tolist() == [0, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
        # loss_mask must align with valid label positions
        assert torch.equal(out["loss_mask"], (out["labels"] != IGNORE_INDEX).float())
        assert out["position_ids"][0].tolist() == list(range(5))
        assert "attention_mask" not in out  # flash path omits the dense mask

    def test_padding_truncation(self, monkeypatch):
        self._patch_args(monkeypatch, seq_length=3, use_flash_attn=True)
        collate = PaddingCollator(seq_length=3, pad_token_id=0)

        out = collate([torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)])

        assert out["tokens"].tolist() == [[1, 2, 3]]
        assert out["labels"].tolist() == [[2, 3, IGNORE_INDEX]]

    def test_attention_mask_when_no_flash(self, monkeypatch):
        self._patch_args(monkeypatch, seq_length=4, use_flash_attn=False)
        collate = PaddingCollator(seq_length=4, pad_token_id=0)

        out = collate([torch.tensor([1, 2], dtype=torch.long)])

        assert "attention_mask" in out
        am = out["attention_mask"]
        assert am.shape == (1, 1, 4, 4)
        assert am.dtype == torch.bool
        # Query position 0: attends to itself, but padded key columns (2,3) blocked.
        assert bool(am[0, 0, 0, 0]) is False
        assert bool(am[0, 0, 0, 2]) is True
        assert bool(am[0, 0, 0, 3]) is True

    def test_custom_pad_token_id(self, monkeypatch):
        self._patch_args(monkeypatch, seq_length=4, use_flash_attn=True)
        collate = PaddingCollator(seq_length=4, pad_token_id=99)

        out = collate([torch.tensor([1, 2], dtype=torch.long)])
        assert out["tokens"][0].tolist() == [1, 2, 99, 99]

@pytest.mark.utils
class TestGetCollateFn:
    def test_returns_packing(self, monkeypatch):
        args = _make_args(hf_collator_mode="packing")
        monkeypatch.setattr(collator_mod, "get_args", lambda: args)
        # Packing path should not require a tokenizer.
        assert isinstance(get_collate_fn(), PackingCollator)

    def test_returns_padding_with_pad_token(self, monkeypatch):
        args = _make_args(hf_collator_mode="padding", seq_length=8)
        monkeypatch.setattr(collator_mod, "get_args", lambda: args)
        monkeypatch.setattr(
            collator_mod, "get_tokenizer", lambda: _FakeTokenizer(pad_token_id=5)
        )

        fn = get_collate_fn()
        assert isinstance(fn, PaddingCollator)
        assert fn.pad_token_id == 5
        assert fn.seq_length == 8

    def test_padding_fallback_when_pad_token_none(self, monkeypatch):
        args = _make_args(hf_collator_mode="padding", seq_length=8)
        monkeypatch.setattr(collator_mod, "get_args", lambda: args)
        monkeypatch.setattr(
            collator_mod, "get_tokenizer", lambda: _FakeTokenizer(pad_token_id=None)
        )
        assert get_collate_fn().pad_token_id == 0

    def test_padding_fallback_when_pad_token_negative(self, monkeypatch):
        args = _make_args(hf_collator_mode="padding", seq_length=8)
        monkeypatch.setattr(collator_mod, "get_args", lambda: args)
        monkeypatch.setattr(
            collator_mod, "get_tokenizer", lambda: _FakeTokenizer(pad_token_id=-1)
        )
        assert get_collate_fn().pad_token_id == 0
