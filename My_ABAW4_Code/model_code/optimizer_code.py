import torch
from torch import nn

def get_optimizer(args, model):
    if args.optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.lr)