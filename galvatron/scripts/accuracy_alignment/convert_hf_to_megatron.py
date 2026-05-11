#!/usr/bin/env python3
"""
Convert HuggingFace-style raw datasets (json/jsonl/csv/parquet/arrow) to
Megatron indexed dataset format (.bin/.idx).
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

INDEX_HEADER = b"MMIDIDX\x00\x00"
DTYPE_CODE_INT32 = 4


def get_bin_path(path_prefix: str) -> str:
    return path_prefix + ".bin"


def get_idx_path(path_prefix: str) -> str:
    return path_prefix + ".idx"


def get_data_files(data_paths: Union[str, List[str]]):
    data_files = []
    if isinstance(data_paths, str):
        data_paths = [data_paths]

    for data_path in data_paths:
        p = Path(data_path)
        if p.is_dir():
            data_files.extend([str(file_path) for file_path in p.iterdir() if file_path.is_file()])
        elif p.is_file():
            data_files.append(str(p))
        else:
            raise FileNotFoundError(f"Dataset {p} not exists.")
    if not data_files:
        raise ValueError(f"No data files found for data_paths={data_paths!r}")

    file_extension = Path(data_files[0]).suffix.lstrip(".").lower()
    if file_extension == "jsonl":
        file_extension = "json"
    if file_extension not in ["parquet", "json", "csv", "arrow"]:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    return data_files, file_extension


def get_text_from_example(example: dict, text_keys: Union[str, List[str]]):
    if isinstance(text_keys, str):
        return example.get(text_keys)
    for key in text_keys:
        if key in example:
            return example[key]
    return None


def tokenize_text(text: str, tokenizer, eos_token_id: Optional[int]) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if eos_token_id is not None:
        ids = ids + [eos_token_id]
    return ids


def split_into_chunks(token_ids: List[int], chunk_len: int):
    chunks = []
    for i in range(0, len(token_ids), chunk_len):
        chunk = token_ids[i : i + chunk_len]
        if chunk:
            chunks.append(chunk)
    return chunks


class IndexedDatasetBuilderLite:
    def __init__(self, bin_path: str):
        self.data_file = open(bin_path, "wb")
        self.sequence_lengths: List[int] = []
        self.document_indices: List[int] = [0]

    def add_item(self, tensor: torch.Tensor) -> None:
        np_array = np.array(tensor.numpy(), dtype=np.int32)
        self.data_file.write(np_array.tobytes(order="C"))
        self.sequence_lengths.append(np_array.size)

    def end_document(self) -> None:
        self.document_indices.append(len(self.sequence_lengths))

    def finalize(self, idx_path: str) -> None:
        self.data_file.close()
        self._write_index(idx_path)

    def _write_index(self, idx_path: str) -> None:
        sequence_pointers = self._sequence_pointers(self.sequence_lengths)
        with open(idx_path, "wb") as f:
            f.write(INDEX_HEADER)
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<B", DTYPE_CODE_INT32))
            f.write(struct.pack("<Q", len(self.sequence_lengths)))
            f.write(struct.pack("<Q", len(self.document_indices)))
            f.write(np.array(self.sequence_lengths, dtype=np.int32).tobytes(order="C"))
            f.write(np.array(sequence_pointers, dtype=np.int64).tobytes(order="C"))
            f.write(np.array(self.document_indices, dtype=np.int64).tobytes(order="C"))

    @staticmethod
    def _sequence_pointers(sequence_lengths: List[int]) -> List[int]:
        curr_ptr = 0
        pointers = []
        for length in sequence_lengths:
            pointers.append(curr_ptr)
            curr_ptr += length * np.dtype(np.int32).itemsize
        return pointers


def parse_text_keys(raw: str) -> Union[str, List[str]]:
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise ValueError("text_keys must not be empty")
    return keys[0] if len(keys) == 1 else keys


def iterate_examples(ds) -> Iterable[dict]:
    for item in ds:
        yield item


def convert_dataset(
    input_path: str,
    tokenizer_model: str,
    output_prefix: str,
    text_keys: Union[str, List[str]] = "text",
    split: str = "train",
    seq_length: int = 8192,
    max_examples: int = 0,
) -> tuple[int, int]:
    data_files, file_extension = get_data_files(input_path)
    dataset = load_dataset(file_extension, data_files=data_files, split=split, streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id

    out_prefix = Path(output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    builder = IndexedDatasetBuilderLite(get_bin_path(str(out_prefix)))

    docs_written = 0
    seqs_written = 0

    for idx, example in enumerate(iterate_examples(dataset), start=1):
        if max_examples > 0 and idx > max_examples:
            break

        text = get_text_from_example(example, text_keys)
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        if not text:
            continue

        token_ids = tokenize_text(text, tokenizer, eos_token_id)
        chunks = split_into_chunks(token_ids, seq_length)
        if not chunks:
            continue

        for chunk in chunks:
            builder.add_item(torch.tensor(chunk, dtype=torch.int32))
            seqs_written += 1
        builder.end_document()
        docs_written += 1

        if docs_written % 10000 == 0:
            print(f"[progress] docs={docs_written} seqs={seqs_written}")

    builder.finalize(get_idx_path(str(out_prefix)))
    return docs_written, seqs_written


def build_default_output_prefix(input_path: str) -> str:
    p = Path(input_path)
    name = p.name if p.name else p.parent.name
    return str(p.parent / f"{name}_text_document")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace dataset to Megatron indexed dataset")
    parser.add_argument("--input-path", required=True, help="HF dataset path (file or directory)")
    parser.add_argument("--tokenizer-model", required=True, help="Tokenizer path or HF repo id")
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Output prefix path. Will create <prefix>.bin and <prefix>.idx",
    )
    parser.add_argument(
        "--text-keys",
        default="text",
        help="Text field(s), comma separated. Example: text,content,raw_text",
    )
    parser.add_argument("--split", default="train", help="Dataset split name")
    parser.add_argument("--seq-length", type=int, default=8192, help="Chunk length")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit number of examples (0 means all)")
    args = parser.parse_args()

    output_prefix = args.output_prefix.strip() or build_default_output_prefix(args.input_path)
    text_keys = parse_text_keys(args.text_keys)

    docs_written, seqs_written = convert_dataset(
        input_path=args.input_path,
        tokenizer_model=args.tokenizer_model,
        output_prefix=output_prefix,
        text_keys=text_keys,
        split=args.split,
        seq_length=args.seq_length,
        max_examples=args.max_examples,
    )

    print(f"[done] docs={docs_written} seqs={seqs_written}")
    print(f"[done] bin={get_bin_path(output_prefix)}")
    print(f"[done] idx={get_idx_path(output_prefix)}")


if __name__ == "__main__":
    main()
