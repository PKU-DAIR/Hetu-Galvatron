from functools import partial
from typing import List

import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
from megatron.legacy.data.vit_dataset import build_train_valid_datasets
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_tp_rank,
    get_ltor_masks_and_position_ids,
)
from torch import Tensor
from torch.utils.data import Dataset

from galvatron.core.runtime.hybrid_parallel_config import get_chunks
from galvatron.core.runtime.pipeline.utils import chunk_batch


def vit_collate_fn(batch):
    pixel_values, labels = zip(*batch)
    pixel_values_ = torch.stack(pixel_values, dim=0)
    labels_ = torch.stack(labels, dim=0)
    return pixel_values_, {"labels": labels_}, None


class DataLoaderForViT(Dataset):
    def __init__(self, config, device):
        self.num_labels = config.num_labels
        self.num_channels = config.num_channels
        self.image_size = config.image_size
        self.dataset_size = 256 * 16 
        self.device = device

        self.pixel_values, self.labels = [], []
        for i in range(self.dataset_size):
            pixel_values = np.random.rand(self.num_channels, self.image_size, self.image_size)
            label = np.random.randint(0, self.num_labels, (1,))
            
            self.pixel_values.append(pixel_values)
            self.labels.append(label)
        
        self.pixel_values = np.array(self.pixel_values)
        self.labels = np.array(self.labels)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        pixel_values = torch.Tensor(self.pixel_values[idx]).to(self.device)
        labels = torch.LongTensor(self.labels[idx]).to(self.device)
        return pixel_values, labels


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    args.dataloader_type = "cyclic"
    args.vision_pretraining = True
    if args.mixed_precision == "fp16":
        args.fp16 = True
    elif args.mixed_precision == "bf16":
        args.bf16 = True
    else:
        args.fp16 = False
        args.bf16 = False
        print_rank_0(f"Warning: Using full precision training with mixed_precision={args.mixed_precision}")

    print_rank_0("> building train, validation, and test datasets for ViT...")
    
    train_ds, valid_ds = build_train_valid_datasets(
        data_path=args.data_path,
        image_size=(args.image_size, args.image_size)
    )
    
    print_rank_0("> finished creating ViT datasets...")

    return train_ds, valid_ds, None


def get_train_valid_test_data_iterators():
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )
    return train_data_iterator, valid_data_iterator, test_data_iterator


def fake_tensor(bsz):
    args = get_args()
    return torch.zeros([bsz, args.num_channels, args.image_size, args.image_size], device="cuda")


def get_batch(data_iterator):
    args = get_args()
    batch_size = args.global_batch_size // mpu.get_data_parallel_world_size()
    
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return fake_tensor(batch_size), {}, None

    def _broadcast(item):
       if item is not None:
           torch.distributed.broadcast(
               item, 
               mpu.get_tensor_model_parallel_src_rank(), 
               group=mpu.get_tensor_model_parallel_group()
           )

    if mpu.get_tensor_model_parallel_rank() == 0:
        data = next(data_iterator)
        
        batch = {
            'pixel_values': data[0].cuda(non_blocking=True), 
            'labels': data[1].cuda(non_blocking=True),        
        }

        if args.pipeline_model_parallel_size == 1 or mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            _broadcast(batch['pixel_values'])
            _broadcast(batch['labels'])
    else:
        dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32
        pixel_values = torch.empty(
            (args.micro_batch_size, args.num_channels, args.image_size, args.image_size), 
            dtype=dtype, 
            device=torch.cuda.current_device()
        )
        labels = torch.empty(
            (args.micro_batch_size,), 
            dtype=torch.int64, 
            device=torch.cuda.current_device()
        )

        if args.pipeline_model_parallel_size == 1 or mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
            _broadcast(pixel_values)
            _broadcast(labels)

        batch = {
            'pixel_values': pixel_values,
            'labels': labels,
        }

    if batch["pixel_values"] is None:
        batch["pixel_values"] = fake_tensor(batch_size)
    
    return (
        batch["pixel_values"], 
        {
            "labels": batch["labels"], 
        },
        loss_func,  
    )


def loss_func(labels, output_tensor):
    args = get_args()
    output_tensor = output_tensor[0]
    losses = output_tensor.float()
    
    loss = torch.sum(losses.view(-1)) / losses.numel()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    
    return loss, averaged_loss[0]