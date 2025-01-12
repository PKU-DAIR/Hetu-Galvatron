import torch
from torch.utils.data import Dataset
import numpy as np
from functools import partial
from galvatron.site_package.megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from galvatron.site_package.megatron.core.datasets.bert_dataset import BERTMaskedWordPieceDataset, BERTMaskedWordPieceDatasetConfig
from galvatron.site_package.megatron.core import mpu, tensor_parallel
from galvatron.site_package.megatron.training import print_rank_0, get_args
from galvatron.site_package.megatron.training.training import build_train_valid_test_data_iterators
from torch import Tensor
from typing import List
from galvatron.site_package.megatron.training import get_tokenizer
from galvatron.site_package.megatron.training.utils import (
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from galvatron.core.hybrid_parallel_config import get_chunks
from galvatron.core.pipeline.utils import chunk_batch

def test_collate_fn(batch):
    tokens = torch.stack(batch, dim=0)
    attention_mask = torch.ones_like(tokens)
    token_type_ids = torch.zeros_like(tokens)
    masked_lm_labels = tokens.clone()
    next_sentence_label = torch.zeros(tokens.size(0), dtype=torch.long)
    
    return tokens, {
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "masked_lm_labels": masked_lm_labels,
        "next_sentence_label": next_sentence_label
    }, None

class DataLoaderForBert(Dataset):
    def __init__(self, args, device):
        self.vocab_size = args.vocab_size
        self.seq_length = args.seq_length
        self.dataset_size = 2560 * 16
        self.device = device
        
        self.mask_token_id = 103
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        
        self.input_ids = []
        self.masked_lm_labels = []
        self.next_sentence_label = []
        
        for i in range(self.dataset_size):
            seq1_len = np.random.randint(1, self.seq_length//2)
            seq2_len = np.random.randint(1, self.seq_length - seq1_len - 3) 
            
            seq1 = np.random.randint(0, self.vocab_size, (seq1_len,))
            seq2 = np.random.randint(0, self.vocab_size, (seq2_len,))
            
            input_seq = self._build_input_sequence(seq1, seq2)
            masked_seq, masked_labels = self._apply_mlm(input_seq.copy())
            
            self.input_ids.append(masked_seq)
            self.masked_lm_labels.append(masked_labels)
            self.next_sentence_label.append(np.random.randint(0, 2))
            
        self.input_ids = np.array(self.input_ids)
        self.masked_lm_labels = np.array(self.masked_lm_labels)
        self.next_sentence_label = np.array(self.next_sentence_label)

    def _build_input_sequence(self, seq1, seq2):
        input_seq = np.zeros(self.seq_length, dtype=np.int64)
        pos = 0
        input_seq[pos] = self.cls_token_id; pos += 1  
        input_seq[pos:pos+len(seq1)] = seq1; pos += len(seq1)
        input_seq[pos] = self.sep_token_id; pos += 1  
        input_seq[pos:pos+len(seq2)] = seq2; pos += len(seq2)
        input_seq[pos] = self.sep_token_id  
        return input_seq

    def _apply_mlm(self, seq):
        masked_labels = np.zeros_like(seq)
        for i in range(len(seq)):
            if seq[i] > 0:  
                if np.random.random() < 0.15:
                    masked_labels[i] = seq[i]
                    if np.random.random() < 0.8:
                        seq[i] = self.mask_token_id
                    elif np.random.random() < 0.5:
                        seq[i] = np.random.randint(0, self.vocab_size)
        return seq, masked_labels

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        return (
            torch.LongTensor(self.input_ids[idx]).to(self.device),
            torch.LongTensor(self.masked_lm_labels[idx]).to(self.device),
            torch.LongTensor([self.next_sentence_label[idx]]).to(self.device)
        )

def test_collate_fn(batch):
    input_ids = torch.stack([item[0] for item in batch])
    masked_lm_labels = torch.stack([item[1] for item in batch])
    next_sentence_labels = torch.stack([item[2] for item in batch])
    
    args = get_args()
    attention_mask = torch.ones_like(input_ids)
    
    return input_ids, {
        "attention_mask": attention_mask,
        "labels": masked_lm_labels,
        "next_sentence_label": next_sentence_labels
    }, None

def get_batch(data_iterator, args):
    batch_size = args.global_train_batch_size // mpu.get_data_parallel_world_size()
    
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return fake_tensor(batch_size), {}, None
    
    batch = get_batch_on_this_tp_rank(data_iterator)
    micro_lossmask = chunk_batch([batch["loss_mask"]], get_chunks(args))
    
    return batch["text"], {
        "attention_mask": batch["padding_mask"],
        "labels": batch["labels"],
        "next_sentence_label": batch["is_random"]
    }, partial(loss_func, micro_lossmask)

def loss_func(micro_lossmask, label, output_tensor):
    loss_mask = micro_lossmask[0][0]
    args = get_args()
    
    mlm_loss = output_tensor[0].float()
    nsp_loss = output_tensor[1].float()
    
    loss_mask = loss_mask.view(-1).float()
    total_loss = mlm_loss * loss_mask + nsp_loss
    loss = torch.sum(total_loss) / loss_mask.sum()
    
    averaged_loss = average_losses_across_data_parallel_group([loss])
    
    micro_lossmask.pop(0)
    return loss, averaged_loss[0]
