
import pandas as pd
import numpy as np
import random
import torch
import sys
import os
from torch import nn
import json
import datetime
import warnings
import math

# 设置随机数种子
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

#创建文件夹
def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

#print到终端与file 调用sys.stdout=Logger(path)
class Logger(object):
    def __init__(self,log_file='log_file.log') -> None:
        self.terminal = sys.stdout
        self.file = open(log_file,'w')
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()
    def reset_path(self, log_file, mode='w'):
        self.file = open(log_file, mode)
    def flush(self):
        self.file.flush()

#计算分类准确数量
def get_classification_count(y_hat, y):
    y_hat = y_hat.argmax(dim=1) # y_hat (batch,class_nums) -> (batch,)
    cmp = y_hat.type(y.dtype) == y
    count = cmp.type(y.dtype).sum()
    return count

def set_parallel_devices(devices, model):
    devices_count = devices.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = devices
    if len(devices_count) > 1:
        model = nn.DataParallel(model)

def get_parameters(size):
    ans = torch.nn.Parameter(torch.empty(size))
    torch.nn.init.xavier_normal_(ans)
    return ans

def get_time_dir():
    curr_time = datetime.datetime.now()
    timestamp = datetime.datetime.strftime(curr_time,'%Y%m%d-%H-%M-%S')
    return timestamp

def set_optimizer_lr(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr

def read_txt_for_lines(txt_path):
    f = open(txt_path, 'r')
    ans = f.readlines()
    f.close()
    return ans

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.train_epoch_nums * len(loader)
    base_lr = args.lr #* args.batch_size / 256

    warmup_steps = 10 * len(loader)
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_json_data(file_path):
    with open(file_path, 'r', encoding = 'utf-8') as f:
        ans = json.load(f)
    f.close()
    return ans

def save_model_config(model, optimizer, epoch, args, log_path, name):
    torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            },
            os.path.join(log_path, name)
        )