import torch
from typing import List, Union, Optional
from torch.utils.data import IterableDataset, Dataset
from datasets import interleave_datasets, load_dataset
from galvatron.core.runtime.parallel_state import get_args, get_tokenizer
from galvatron.core.runtime.datasets.huggingface.utils import (
    get_data_files,
    get_text_from_example,
    tokenize_text,
    split_into_chunks,
)
from galvatron.core.runtime.parallel_state import (
    get_vocab_data_parallel_world_size as get_data_parallel_world_size,
    get_vocab_data_parallel_rank as get_data_parallel_rank,
)

FIXED_DATA_SHARD_COUNT = 8

class HFDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_seq_len: int,
        text_keys: Union[str, List[str]] = "text",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_keys = text_keys
        try:
            self.eos_token_id = tokenizer.eod
        except Exception as e:
            print(f"Failed to get eos_token_id from tokenizer: {e}")

        self._chunks = self.preprocess()

    def preprocess(self):
        if self.data is None:
            return

        def _map_fn(examples):
            chunks = []
            num_examples = len(next(iter(examples.values()))) if examples else 0
            for i in range(num_examples):
                example_i = {key: value[i] for key, value in examples.items()}
                text = get_text_from_example(example_i, self.text_keys)
                if text is None:
                    continue
                token_ids = tokenize_text(text, self.tokenizer, self.eos_token_id)
                chunks.extend(split_into_chunks(token_ids, self.max_seq_len))
            return {"input_ids": chunks}

        # remove all original columns and keep only `input_ids`.
        chunks = self.data.map(
            _map_fn,
            batched=True,
            remove_columns=self.data.column_names,
            desc="tokenize and split to chunks",
        )
        return chunks
    
    def __len__(self) -> int:
        if hasattr(self, "_chunks") and self._chunks is not None:
            return len(self._chunks)
        return 0

    def __getitem__(self, index: int) -> torch.Tensor:
        if not hasattr(self, "_chunks") or self._chunks is None:
            raise RuntimeError("HFDataset has no data.")

        example = self._chunks[index]
        input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
        return input_ids

class HFIterableDataset(IterableDataset):
    def __init__(
        self,
        data,
        tokenizer,
        max_seq_len,
        text_keys,
        device
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_keys = text_keys
        self.device = device
        try:
            self.eos_token_id = tokenizer.eod
        except Exception as e:
            print(f"Failed to get eos_token_id from tokenizer: {e}")

    def __iter__(self):
         for example in self.data:
            text = get_text_from_example(example, self.text_keys)
            if text is None:
                continue
            token_ids = tokenize_text(text, self.tokenizer, self.eos_token_id)
            chunks = split_into_chunks(token_ids, self.max_seq_len)
            for chunk in chunks:
                yield torch.tensor(chunk, dtype=torch.long)


class HFStreamingShardDataset(IterableDataset):
    """Round-robin shard dataset over a streaming HF dataset."""

    def __init__(self, data, dp_rank: int, dp_world_size: int):
        self.data = data
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

    def __iter__(self):
        for idx, example in enumerate(self.data):
            if idx % self.dp_world_size == self.dp_rank:
                yield example


def build_hf_iterable_dataset(
    train_ds_path: str,
    split: str = "train",
    text_keys: Union[str, List[str]] = "text",
    device: Optional[torch.device] = None,
    shuffle_buffer_size: int = 0,
    seed: int = 42,
):
    data_files, file_extension = get_data_files(train_ds_path)
    args = get_args()

    dataset = load_dataset(file_extension, data_files=data_files, split=split, streaming=True)
    if split == "train" and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)


    dp_world_size = get_data_parallel_world_size()
    if dp_world_size > 1:
        dp_rank = get_data_parallel_rank()
        # shard by files and by examples, not only by examples
        dataset = dataset.shard(num_shards=dp_world_size, index=dp_rank)

    return HFIterableDataset(
        data=dataset,
        tokenizer=get_tokenizer(),
        max_seq_len=args.train.seq_length,
        text_keys=text_keys,
        device=device,
    )

def build_hf_dataset(
    train_ds_path: str,
    text_keys: Union[str, List[str]] = "text",
    split: str = "train",
) -> HFDataset:

    data_files, file_extension = get_data_files(train_ds_path)

    dataset = None

    dataset = load_dataset(file_extension, data_files=data_files, split=split)
    args = get_args()
    return HFDataset(
        data=dataset,
        tokenizer=get_tokenizer(),
        max_seq_len=args.train.seq_length,
        text_keys=text_keys,
    )


def build_hf_streaming_dataset(
    train_ds_path: str,
    split: str = "train",
    shuffle_buffer_size: int = 0,
    seed: int = 42,
    dp_rank: Optional[int] = None,
    dp_world_size: Optional[int] = None,
):
    data_files, file_extension = get_data_files(train_ds_path)
    dataset = load_dataset(file_extension, data_files=data_files, split=split, streaming=True)

    if split == "train" and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    if dp_rank is None or dp_world_size is None:
        return dataset

    # Single local file can fail inside dataset.shard due to empty gen_kwargs_list
    if len(data_files) == 1:
        return HFStreamingShardDataset(
            data=dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
        )

    shard_count = FIXED_DATA_SHARD_COUNT
    if dp_world_size > shard_count:
        raise ValueError(
            f"dp_world_size ({dp_world_size}) > FIXED_DATA_SHARD_COUNT ({shard_count})"
        )
    all_shards = [
        dataset.shard(num_shards=shard_count, index=i, contiguous=True)
        for i in range(shard_count)
    ]
    local_indices = [i for i in range(shard_count) if i % dp_world_size == dp_rank]
    local_shards = [all_shards[i] for i in local_indices]

    return interleave_datasets(local_shards, stopping_strategy="all_exhausted")