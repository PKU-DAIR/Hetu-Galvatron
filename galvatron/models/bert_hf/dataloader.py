import os
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import numpy as np
from typing import List
from megatron.training.utils import average_losses_across_data_parallel_group, get_batch_on_this_tp_rank
from megatron.training import get_args
from megatron.core import mpu, tensor_parallel
from functools import partial
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.bert_dataset import (
    BERTMaskedWordPieceDataset,
    BERTMaskedWordPieceDatasetConfig
)
from megatron.core import tensor_parallel
from megatron.training import print_rank_0, get_args
from megatron.training.training import build_train_valid_test_data_iterators
from megatron.training import get_tokenizer
<<<<<<< HEAD
from galvatron.core.runtime.hybrid_parallel_config import get_chunks
from galvatron.core.runtime.pipeline.utils import chunk_batch
=======
from galvatron.core.hybrid_parallel_config import get_chunks
from galvatron.core.runtime.pipeline.utils import chunk_batch
>>>>>>> 0e554e6502dab21f2e27e26454504bed37ac6828

class DataLoaderForBert(Dataset):
    def __init__(self, args, device, dataset_size=64*16):
        self.vocab_size = args.vocab_size
        self.seq_length = args.seq_length
        self.dataset_size = dataset_size
        
        min_tokens_per_sentence = 10  
        special_tokens_count = 3     
        min_total_length = min_tokens_per_sentence * 2 + special_tokens_count
        
        self.seq_lengths = np.random.randint(min_total_length, self.seq_length+1, (self.dataset_size,))
        
        self.device = device
        
        self.cls_id = 101  
        self.sep_id = 102  
        self.pad_id = 0    
        self.mask_id = 103 
        
        self.input_ids = []
        self.attention_masks = []
        self.token_type_ids = []
        self.mlm_labels = []
        self.nsp_labels = []
        
        self.special_token_ids = {
            self.cls_id,    
            self.sep_id,    
            self.pad_id,   
            self.mask_id,  
        }
        self.valid_ids = torch.tensor(
            [i for i in range(self.vocab_size) if i not in self.special_token_ids],
            dtype=torch.long
        )

        for i in range(self.dataset_size):
            total_length = self.seq_lengths[i]

            available_length = total_length - special_tokens_count
            
            max_length_a = available_length - min_tokens_per_sentence
            min_length_a = min_tokens_per_sentence
            seq_length_a = np.random.randint(min_length_a, max_length_a + 1)
            
            seq_length_b = available_length - seq_length_a
            
            sentence_a = self.create_random_sentence(seq_length_a)
            sentence_b = self.create_random_sentence(seq_length_b)
            
            is_next_random = np.random.random() < 0.5 
            # [CLS] sentence_a [SEP] sentence_b [SEP] [PAD] [PAD] ...
            tokens = [self.cls_id] + sentence_a + [self.sep_id] + sentence_b + [self.sep_id]
            
            padding_length = self.seq_length - len(tokens)
            tokens.extend([self.pad_id] * padding_length)
            
            attention_mask = [1] * (len(sentence_a) + len(sentence_b) + 3) + [0] * padding_length
            
            token_type_ids = [0] * (len(sentence_a) + 2) + [1] * (len(sentence_b) + 1) + [0] * padding_length
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            masked_tokens, mlm_labels = self.create_masked_tokens(tokens.clone(), np.random.RandomState(i))
            
            self.input_ids.append(masked_tokens.numpy())
            self.attention_masks.append(np.array(attention_mask))
            self.token_type_ids.append(np.array(token_type_ids))
            self.mlm_labels.append(mlm_labels.numpy())
            self.nsp_labels.append(1 if is_next_random else 0)
        
        self.input_ids = np.array(self.input_ids)
        self.attention_masks = np.array(self.attention_masks)
        self.token_type_ids = np.array(self.token_type_ids)
        self.mlm_labels = np.array(self.mlm_labels)
        self.nsp_labels = np.array(self.nsp_labels)
        
    def create_random_sentence(self, length):
        indices = torch.randint(0, len(self.valid_ids), (length,))
        return self.valid_ids[indices].tolist()
        
    def create_masked_tokens(self, tokens, numpy_rng):
        labels = tokens.clone()

        special_tokens_mask = torch.tensor(
            [1 if x in [self.cls_id, self.sep_id, self.pad_id] else 0 for x in tokens],
            dtype=torch.bool
        )
        
        probability_matrix = torch.full(labels.shape, 0.15)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        labels[~masked_indices] = -100 
        
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        tokens[indices_replaced] = self.mask_id
        
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_indices = torch.randint(0, len(self.valid_ids), labels.shape)
        random_words = self.valid_ids[random_indices]
        tokens[indices_random] = random_words[indices_random]
        
        return tokens, labels
        
    def __len__(self):
        return self.dataset_size
        
    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
            
        return {
            'input_ids': torch.LongTensor(self.input_ids[idx]).to(self.device),
            'attention_mask': torch.BoolTensor(self.attention_masks[idx]).to(self.device),
            'token_type_ids': torch.LongTensor(self.token_type_ids[idx]).to(self.device),
            'mlm_labels': torch.LongTensor(self.mlm_labels[idx]).to(self.device),
            'nsp_labels': torch.LongTensor([self.nsp_labels[idx]]).to(self.device)
        }

def random_collate_fn_bert(batch):
    """Make batch suitable for GalvatronModel
    
    Args:
        batch: List[Dict] - DataLoaderForBert.__getitem__
    Returns:
        tuple: (batch, kwargs, loss_func)
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    mlm_labels = torch.stack([item['mlm_labels'] for item in batch])
    
    batch = [input_ids]
    
    kwargs = {
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": mlm_labels }
    
    return batch, kwargs, None 

def is_dataset_built_on_rank():
    is_first = mpu.is_pipeline_first_stage()
    is_last = mpu.is_pipeline_last_stage()
    tensor_rank = mpu.get_tensor_model_parallel_rank()
    should_build = (is_first or is_last) and tensor_rank == 0
    return should_build

def core_bert_dataset_config_from_args(args):
    """Create dataset config from args."""
    tokenizer = get_tokenizer()

    if isinstance(args.data_path, list):
        args.data_path = args.data_path[0]

    masking_config = {
        "masking_probability": 0.15,        
        "short_sequence_probability": 0.1,  
        "masking_max_ngram": 1,            
        "masking_do_full_word": True,      
        "masking_do_permutation": False,    
        "masking_use_longer_ngrams": False, 
        "masking_use_geometric_distribution": False,
        "classification_head": True 
    }
    
    if hasattr(args, "mask_prob"):
        masking_config["masking_probability"] = args.mask_prob
    if hasattr(args, "short_seq_prob"):
        masking_config["short_sequence_probability"] = args.short_seq_prob
    
    return BERTMaskedWordPieceDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=[args.data_path],
        blend_per_split=None,
        split=args.split,
        path_to_cache=args.data_cache_path,
        mock=args.mock_data,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
       
        **masking_config
    )

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, validation and test datasets for BERT."""
    args = get_args()
    print_rank_0("> building train, validation, and test datasets for BERT...")
    
    config = core_bert_dataset_config_from_args(args)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        BERTMaskedWordPieceDataset,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config  
    ).build()
    print_rank_0("> finished creating BERT datasets...")

    return train_ds, valid_ds, test_ds

def get_train_valid_test_data_iterators():
    """Build train, valid, and test data iterators."""
    train_valid_test_datasets_provider.is_distributed = True
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        train_valid_test_datasets_provider
    )
    return train_data_iterator, valid_data_iterator, test_data_iterator


def fake_tensor(bsz):
    return torch.zeros([bsz, 1], device="cuda")


def get_batch(data_iterator):
    args = get_args()
    batch_size = args.global_train_batch_size // mpu.get_data_parallel_world_size()
    
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return fake_tensor(batch_size), {}, None
    
    batch = get_batch_on_this_tp_rank(data_iterator)

    micro_lossmask = chunk_batch([batch["loss_mask"]], get_chunks(args))
    
    if batch["tokens"] == None:
        batch["tokens"] = fake_tensor(batch_size)

    return batch["text"], {
        "attention_mask": batch["padding_mask"],
        "token_type_ids": batch["types"],
        "labels": batch["labels"],
    }, partial(loss_func, micro_lossmask)

    
def loss_func(micro_lossmask: List[Tensor], label: List, output_tensor: List):
    loss_mask = micro_lossmask[0][0]
    mlm_loss = output_tensor[0] 
    mlm_loss = mlm_loss.float()
    loss_mask = loss_mask.view(-1).float()
    masked_mlm_loss = torch.sum(mlm_loss.view(-1) * loss_mask) / loss_mask.sum()

    averaged_loss = average_losses_across_data_parallel_group([masked_mlm_loss])

    micro_lossmask.pop(0)
    return masked_mlm_loss, averaged_loss[0]