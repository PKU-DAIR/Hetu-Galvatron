import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import BertConfig, BertForPreTraining
from tqdm import tqdm
from galvatron.utils import set_seed, print_loss
from galvatron.core import initialize_galvatron, GalvatronProfiler
from galvatron.models.bert_hf.meta_configs import config_from_meta, set_model_config
from galvatron.models.bert_hf.arguments import model_args
from galvatron.models.bert_hf.dataloader import DataLoaderForBert

def model_forward(model, batch):
    """Forward pass for BERT pre-training"""
    outputs = model(
        input_ids=batch["text"],
        attention_mask=batch["padding_mask"],
        token_type_ids=batch["types"],
        labels=batch["labels"],
        next_sentence_label=batch["is_random"]
    )
    return outputs.loss

def train(args):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")

    config = config_from_meta(args.model_size)
    config = set_model_config(config, args, True)

    print("Creating Model...")
    model = BertForPreTraining(config)
    model.to(device)
    
    print("Creating Dataloader...")
    dataset = DataLoaderForBert(args, device)
    trainloader = DataLoader(
        dataset=dataset,
        batch_size=args.global_train_batch_size,
        shuffle=False
    )
    
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    profiler = GalvatronProfiler(args)
    profiler.set_profiler_single()

    profiler.profile_memory(0, "After creating model")
    print("Start training...")
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            profiler.profile_time_start(iter)
            
            profiler.profile_memory(iter, "Before Forward")

            loss = model_forward(model, batch)

            profiler.profile_memory(iter, "After Forward")

            loss.backward()

            profiler.profile_memory(iter, "After Backward")
            
            optimizer.step()

            profiler.profile_memory(iter, "After optimizer_step")
            
            optimizer.zero_grad()

            print_loss(args, loss, ep, iter)
            
            profiler.post_profile_memory(iter)
            profiler.profile_time_end(iter)

if __name__ == '__main__':
    args = initialize_galvatron(model_args, mode='train')
    set_seed()
    train(args)