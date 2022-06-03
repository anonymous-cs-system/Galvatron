import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import ViTConfig, ViTForImageClassification
import argparse
from tqdm import tqdm
import numpy as np
import random
import os
import time
from data import build_dataset
from config import get_config

import sys
sys.path.insert(0, '..')
from utils import print_peak_memory

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(args, conf):
    cuda_condition = torch.cuda.is_available()
    device = torch.device("cuda:%d"%args.gpu_id if cuda_condition else "cpu")
    rank = args.gpu_id

    print("Creating Dataloader...")
    dataset, num_classes = build_dataset(is_train=True, config=conf)
    trainloader = DataLoader(dataset=dataset,
                            batch_size=args.train_batch_size,
                            shuffle=False)

    print("Creating Model...")
    config = ViTConfig(hidden_size=args.hidden_size,
                        num_hidden_layers=args.num_hidden_layers, 
                        num_attention_heads=args.num_attention_heads, 
                        intermediate_size=args.hidden_size*4, 
                        attention_probs_dropout_prob=args.dropout_prob,
                        hidden_dropout_prob=args.dropout_prob,
                        image_size=args.image_size,
                        patch_size=args.patch_size,
                        num_channels=args.num_channels)
    config.num_labels = num_classes
    model = ViTForImageClassification(config)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay)

    if args.profile:
        print_peak_memory("After creating model", rank, args.profile_type)

    print("Start training...")
    for ep in range(args.epochs):
        if not args.check_loss and not args.profile:
            trainloader = tqdm(trainloader)
        for iter, batch in enumerate(trainloader):
            start_time = time.time()
            input, label= [tensor.to(device) for tensor in batch]

            if args.profile:
                torch.cuda.reset_peak_memory_stats(rank)
                print_peak_memory("\nBefore Forward", rank, args.profile_type)

            # model forward
            loss = model(pixel_values = input, labels = label).loss

            if args.profile:
                print_peak_memory("After Forward", rank, args.profile_type)

            loss.backward()

            if args.profile:
                print_peak_memory("After Backward", rank, args.profile_type)
            
            optimizer.step()

            if args.profile:
                print_peak_memory("After optimizer_step", rank, args.profile_type)
            
            optimizer.zero_grad()

            end_time = time.time()
            if args.check_loss or args.profile:
                print('[Epoch %d] (Iteration %d): Loss = %.6f, Time = %.3f'% (ep,iter,loss.item(), end_time-start_time))
            
            if args.profile and iter >= 2:
                exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size for single GPU"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--image_size", type=int, default=224, help="Input image size."
    )
    parser.add_argument(
        "--patch_size", type=int, default=16, help="Patch size of ViT."
    )
    parser.add_argument(
        "--num_channels", type=int, default=3, help="Number of channels."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.0, help="Dropout rate."
    )
    parser.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate of adam")
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        '--data-path', default = 'ImageNet', type=str, help='path to dataset'
    )
    parser.add_argument(
        '--zip', type=bool, default=True, help='use zipped dataset instead of folder dataset'
    )
    parser.add_argument(
        '--cache-mode', type=str, default='no', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece'
    )
    parser.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel'
    )
    parser.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    parser.add_argument(
        "--profile_type", type=str, default='allocated', help="Profile allocated memory or reserved memory.",
        choices = ['allocated', 'reserved'],
    )


    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    set_seed()

    train(args, config)