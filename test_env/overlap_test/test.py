import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data import Dataset
import argparse
from tqdm import tqdm
import numpy as np
import random
import time
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def set_seed():
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class PauseFunc(torch.autograd.Function):
    @staticmethod
    def forward():
        time.sleep(0.01)
        return

    @staticmethod
    def backward():
        time.sleep(0.01)
        return

class mlp_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.ModuleList([nn.Linear(4096, 4096, bias=False).cuda() for _ in range(24)])
        self.pausefunc = PauseFunc()

    def forward(self, hidden_states):
        self.pausefunc.forward()# Pause to seperate forward and backward
        for i in range(24):
            hidden_states = self.mlp[i](hidden_states)
        self.pausefunc.forward()# Pause to seperate forward and backward
        return hidden_states

class DataLoaderRandom(Dataset):
    def __init__(self):
        self.dataset_size = args.train_batch_size*10*torch.distributed.get_world_size()
        self.input = np.random.randint(0, 100, size=(self.dataset_size, 4096))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError
        input = torch.FloatTensor(self.input[idx])
        return input

def train(args):
    torch.distributed.init_process_group(backend="nccl")
    local_rank = args.local_rank
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dataset = DataLoaderRandom()
    trainloader = DataLoader(dataset=dataset,
                            batch_size=args.train_batch_size,
                            sampler=DistributedSampler(dataset,shuffle=False))

    print("[Rank %d] Creating Model..."%rank)
    model = mlp_model()
    model.to(device)
    if args.ddp_on:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    print("[Rank %d] Start training..."%rank)
    for input in tqdm(trainloader):
        loss = model(input.to(device)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank" ,type=int,default=-1)
    parser.add_argument(
        "--train_batch_size", type=int, default=32, help="Training batch size for single GPU"
    )
    parser.add_argument(
        "--ddp_on", type=int, default=1, help="Turn on ddp or not."
    )
    args = parser.parse_args()
    set_seed()
    train(args)