"""Generic data loading utilities for causal language model training.

Provides:
- ``CausalLMDataset`` / ``random_collate_fn``: synthetic random data for profiling.
- ``get_train_valid_test_data_iterators``: Megatron blended-dataset pipeline.
- ``get_batch`` / ``loss_func``: micro-batch fetching with loss-mask support.
"""

from functools import partial
from typing import List
import json

import numpy as np
import torch
import random
from torch import Tensor
from torch.utils.data import Dataset

from galvatron.core.runtime.parallel_state import get_args
from galvatron.core.runtime.hybrid_parallel_config import get_chunks
from galvatron.core.runtime.pipeline.utils import chunk_batch
from galvatron.core.runtime.transformer.rotary_pos_embedding import RotaryEmbedding
from galvatron.core.runtime.datasets.megatron.utils import get_blend_from_list
from galvatron.core.runtime import parallel_state
from galvatron.core.runtime.datasets.megatron.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from galvatron.core.runtime.datasets.megatron.gpt_dataset import GPTDataset, GPTDatasetConfig
from galvatron.core.runtime.parallel_state import get_args, get_tokenizer
from galvatron.core.runtime.utils.utils import print_rank_0
from galvatron.core.runtime.utils.rerun_state_machine import RerunDataIterator
from galvatron.core.runtime.utils.utils import get_batch_on_this_tp_rank, get_batch_on_this_cp_rank, average_losses_across_data_parallel_group

# =========================================================================
# Fake data
# =========================================================================

class FakeCausalLMDataset(Dataset):
    """Generate random token sequences for testing / profiling."""

    def __init__(self, args, device, dataset_size=2560 * 16):
        self.vocab_size = args.model.vocab_size
        self.seq_length = args.train.seq_length
        self.dataset_size = dataset_size
        self.device = device
        self.input_ids = np.random.randint(0, self.vocab_size, (dataset_size, self.seq_length + 1))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return torch.LongTensor(self.input_ids[idx]).to(self.device)


def random_collate_fn(batch):
    """Collate for ``CausalLMDataset``: split into tokens / labels, build causal mask."""
    tokens_ = torch.stack(batch, dim=0)
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    args = get_args()
    if not args.train.use_flash_attn:
        seq_length = tokens.size(1)
        attention_mask = torch.tril(
            torch.ones((1, seq_length, seq_length), device=tokens.device)
        ).view(1, 1, seq_length, seq_length)
        attention_mask = attention_mask < 0.5
    else:
        attention_mask = None
    rotary_embedding = _build_batch_rotary_embedding(args)
    return tokens, {"attention_mask": attention_mask, "labels": labels, "rotary_embedding": rotary_embedding}, None


# =========================================================================
# Megatron blended dataset (real data)
# =========================================================================

def build_pretraining_data_loader(dataset, consumed_samples):
    """Build dataloader given an input dataset."""

    if dataset is None:
        return None
    args = get_args().train

    # Megatron sampler
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=parallel_state.get_vocab_dp_rank(),
            data_parallel_size=parallel_state.get_vocab_dp_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            dataset,
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=parallel_state.get_vocab_dp_rank(),
            data_parallel_size=parallel_state.get_vocab_dp_world_size(),
            data_sharding=args.data_sharding)
    elif args.dataloader_type == "external":
        # External dataloaders are passed through. User is expected to provide a
        # torch-compatible dataloader and define samplers, if needed.
        return dataset
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       pin_memory=True,
                                       persistent_workers=True if args.num_workers > 0 else False,
                                       )

class MegatronPretrainingSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class RandomSeedDataset(Dataset):

    def __init__(self, dataset):
        args = get_args()
        self.base_seed = args.train.seed
        self.curr_seed = args.train.seed
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.curr_seed = self.base_seed + epoch

    def __getitem__(self, idx):
        seed = idx + self.curr_seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        return self.dataset[idx]


class MegatronPretrainingRandomSampler:

    def __init__(self, dataset, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size, data_sharding):
        # Keep a copy of input params for later use.
        self.dataset = dataset
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.data_sharding = data_sharding
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.data_parallel_size
            start_idx = self.data_parallel_rank * bucket_size

            g = torch.Generator()
            g.manual_seed(self.epoch)
            random_idx = torch.randperm(bucket_size, generator=g).tolist()
            idx_range = [start_idx + x for x in random_idx[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            g = torch.Generator()
            g.manual_seed(self.epoch)
            idx_range_total = \
                torch.randperm(full_bucket_size, generator=g).tolist()
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.data_parallel_rank::self.data_parallel_size]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []


def get_blend_and_blend_per_split(args):
    """Get blend and blend_per_split from passed-in arguments. Uses args.data for paths/split."""
    data = args.data
    use_data_path = data.data_path is not None or data.data_args_path is not None
    use_per_split_data_path = any(
        elt is not None
        for elt in [data.train_data_path, data.valid_data_path, data.test_data_path]
    ) or data.per_split_data_args_path is not None

    blend = None
    blend_per_split = None
    if use_data_path:
        if data.data_args_path is not None:
            assert data.data_path is None
            with open(data.data_args_path, 'r') as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert data.data_path is not None
            blend = get_blend_from_list(data.data_path)
    elif use_per_split_data_path:
        if data.per_split_data_args_path is not None:
            with open(data.per_split_data_args_path, 'r') as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()
                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"])
                ]
        else:
            blend_per_split = [
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path)
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split


def get_train_valid_test_num_samples():
    """Train/valid/test num samples."""

    args = get_args().train

    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    eval_iters = (args.train_iters // args.eval_interval + 1) * \
                 args.eval_iters
    test_iters = args.eval_iters

    return (

        train_samples,
        eval_iters * args.global_batch_size,
        test_iters * args.global_batch_size,
    )


def build_train_valid_test_datasets(build_train_valid_test_datasets_provider):
    """Build pretraining datasets."""
    train_valid_test_num_samples = get_train_valid_test_num_samples()
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_valid_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_valid_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_valid_test_num_samples[2]))
    return build_train_valid_test_datasets_provider(train_valid_test_num_samples)


def build_train_valid_test_data_loaders(
        build_train_valid_test_datasets_provider):
    """Build pretraining data loaders."""

    args = get_args().train

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')

    # Backward compatibility, assume fixed batch size.
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, \
            'Only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = (args.iteration // args.eval_interval) * \
                args.eval_iters * args.global_batch_size

    # Rely on distributed-aware core datasets, temporary
    is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)

    # Construct the data pipeline
    if is_distributed or parallel_state.get_vocab_tp_sp_rank() == 0:

        # Build datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            build_train_valid_test_datasets_provider)
        # Build dataloders.
        train_dataloader = build_pretraining_data_loader(
            train_ds, args.consumed_train_samples)
        if args.skip_train:
            valid_dataloader = build_pretraining_data_loader(valid_ds, 0)
        else:
            valid_dataloader = build_pretraining_data_loader(
                valid_ds, args.consumed_valid_samples)
        test_dataloader = build_pretraining_data_loader(test_ds, 0)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.tensor(
            [int(do_train), int(do_valid), int(do_test)],
            dtype=torch.long, device='cuda')
    else:
        flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')

    torch.distributed.broadcast(flags, 0)

    args.do_train = getattr(args, "do_train", False) or flags[0].item()
    args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
    args.do_test = getattr(args, "do_test", False) or flags[2].item()

    return train_dataloader, valid_dataloader, test_dataloader


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider):
    """Build pretraining data iterators."""

    args = get_args().train

    # Build loaders.
    train_dataloader, valid_dataloader, test_dataloader = \
        build_train_valid_test_data_loaders(
            build_train_valid_test_datasets_provider)

    # Build iterators.
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic', 'external']

    def cyclic_iter(iter):
        while True:
            for x in iter:
                yield x

    def _get_iterator(dataloader_type, dataloader):
        """Return dataset iterator."""
        if dataloader_type == "single":
            return RerunDataIterator(iter(dataloader))
        elif dataloader_type == "cyclic":
            return RerunDataIterator(iter(cyclic_iter(dataloader)))
        elif dataloader_type == "external":
            # External dataloader is passed through. User is expected to define how to iterate.
            if isinstance(dataloader, list):
                return [RerunDataIterator(d) for d in dataloader]
            else:
                return RerunDataIterator(dataloader)
        else:
            raise RuntimeError("unexpected dataloader type")

    if train_dataloader is not None:
        train_data_iterator = _get_iterator(dl_type, train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = _get_iterator(dl_type, valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = _get_iterator(dl_type, test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def _build_random_data_iterator():
    """Build a cyclic iterator over FakeCausalLMDataset for profiling."""
    args = get_args()
    device = torch.device("cuda", args.local_rank)
    dataset = FakeCausalLMDataset(args, device)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train.micro_batch_size,
        collate_fn=random_collate_fn,
        shuffle=False,
    )
    def _cyclic(loader):
        while True:
            for batch in loader:
                yield batch
    return _cyclic(dataloader)


def get_train_valid_test_data_iterators():
    """Build iterators using Megatron's blended dataset pipeline or random data."""
    args = get_args()

    if getattr(args.data, 'use_random_dataset', False):
        print_rank_0('> using random synthetic dataset for profiling ...')
        train_iter = _build_random_data_iterator()
        return train_iter, None, None

    def _is_dataset_built_on_rank():
        return (
            parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage()
        ) and parallel_state.get_vocab_tp_sp_rank() == 0

    def _datasets_provider(train_val_test_num_samples):
        args = get_args()
        tokenizer = get_tokenizer()
        blend, blend_per_split = get_blend_and_blend_per_split(args)
        ds_config = GPTDatasetConfig(
            random_seed=args.train.seed,
            sequence_length=args.train.seq_length,
            blend=blend,
            blend_per_split=blend_per_split,
            split=args.data.split,
            num_dataset_builder_threads=args.data.num_dataset_builder_threads,
            path_to_cache=args.data.data_cache_path,
            mmap_bin_files=args.data.mmap_bin_files,
            tokenizer=tokenizer,
            reset_position_ids=args.data.reset_position_ids,
            reset_attention_mask=args.data.reset_attention_mask,
            eod_mask_loss=args.data.eod_mask_loss,
            create_attention_mask=args.data.create_attention_mask_in_dataloader,
            s3_cache_path=args.data.s3_cache_path,
        )
        print_rank_0("> building train, validation, and test datasets ...")
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            GPTDataset, train_val_test_num_samples, _is_dataset_built_on_rank, ds_config
        ).build()
        print_rank_0("> finished creating datasets ...")
        return train_ds, valid_ds, test_ds

    _datasets_provider.is_distributed = True
    return build_train_valid_test_data_iterators(_datasets_provider)


# =========================================================================
# Batch construction
# =========================================================================

def _build_batch_rotary_embedding(args):
    """Match laer-moe-test moe/dataloader.py: fixed-seq RoPE passed into every layer."""
    m = args.model
    t = args.train
    head_dim = m.kv_channels or (m.hidden_size // m.num_attention_heads)
    rotary_pos_emb = RotaryEmbedding(
        head_dim,
        m.rotary_percent or 1.0,
        seq_len_interpolation_factor=getattr(m, "rotary_seq_len_interpolation_factor", None),
        rotary_base=m.rotary_base or 10000,
    )
    return rotary_pos_emb(t.seq_length)


def get_batch(data_iterator):
    """Fetch a micro-batch and build the loss function closure."""
    args = get_args()

    if getattr(args.data, 'use_random_dataset', False):
        return next(data_iterator)

    batch_size = args.train.global_batch_size // parallel_state.get_vocab_dp_world_size()

    if (not parallel_state.is_pipeline_first_stage()) and (not parallel_state.is_pipeline_last_stage()):
        return torch.zeros([batch_size, 1], device="cuda"), {}, None

    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)

    micro_lossmask = chunk_batch([batch["loss_mask"]], get_chunks(args))

    tokens = batch.get("tokens")
    if tokens is None:
        tokens = torch.zeros([batch_size, 1], device="cuda").long()

    return (
        tokens,
        {
            "position_ids": batch.get("position_ids"),
            "attention_mask": batch.get("attention_mask"),
            "labels": batch.get("labels"),
            "rotary_embedding": _build_batch_rotary_embedding(args),
        },
        partial(_loss_func, micro_lossmask),
    )


def _loss_func(micro_lossmask, label: List, output_tensor: List):

    loss_mask = micro_lossmask[0][0]
    output_tensor = output_tensor[0]
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    micro_lossmask.pop(0)
    return loss, averaged_loss[0]
