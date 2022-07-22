from torch.utils.data import DataLoader, Dataset
import torch
from PIL import Image
import os
import utils
import numpy as np
from data_code.data_transformers import get_data_transforms

class MyDataset(Dataset):
    def __init__(self, data, set_path, transforms) -> None:
        super().__init__()
        self.data = data
        self.set_path = set_path
        self.transforms = transforms
        self.datalength = len(data)

    @torch.no_grad()
    def __getitem__(self, index):

        data = self.data[index]
        video_name = data[0]

        ans_image, ans_valence, ans_arousal, ans_expression, ans_aus = [], [], [], [], []
        for video_frame in data[1]:
            file_path = os.path.join(self.set_path, video_name, video_frame[0])
            image = Image.open(file_path)
            ans_image.append(self.transforms(image))
            ans_valence.append(video_frame[1])
            ans_arousal.append(video_frame[2])
            ans_expression.append(video_frame[3])
            ans_aus.append(torch.tensor(video_frame[4:]))
        
        return torch.stack(ans_image), torch.tensor(ans_valence), torch.tensor(ans_arousal), torch.tensor(ans_expression), torch.stack(ans_aus)

    def __len__(self):
        return self.datalength


class MytestDataset(Dataset):
    def __init__(self, data, set_path, transforms) -> None:
        super().__init__()
        self.data = data
        self.set_path = set_path
        self.transforms = transforms
        self.datalength = len(data)

    @torch.no_grad()
    def __getitem__(self, index):

        data = self.data[index]
        video_name = data[0]

        ans_image, names = [], []
        for video_frame in data[1]:
            file_path = os.path.join(self.set_path, video_name, video_frame)
            names.append(os.path.join(video_name, video_frame))
            image = Image.open(file_path)
            ans_image.append(self.transforms(image))
            
        
        return names, torch.stack(ans_image)

    def __len__(self):
        return self.datalength

def get_train_data(train_json_data, min_time_step, max_time_step, forward_time_step):
    data = []
    for key in train_json_data.keys():
        if min_time_step < len(train_json_data[key]) < max_time_step:
            data.append([key, train_json_data[key]])
        elif max_time_step < len(train_json_data[key]):
            index = max_time_step
            while index < len(train_json_data[key]):
                data.append([key, train_json_data[key][index-max_time_step:index]])
                index = min(index+forward_time_step, len(train_json_data[key]))
    return data

def get_valid_data(valid_json_data):
    data = []
    for key in valid_json_data.keys():
        data.append([key, valid_json_data[key]])
    return data

def get_task_train_data(args, task):
    train_transformes, valid_transformes = get_data_transforms(args)
    task_json = {
        'AU': 'data_vedio_dict/train_dict_AU.json',
        'EXPR': 'data_vedio_dict/train_dict_EXPR.json',
        'VA': 'data_vedio_dict/train_dict_VA.json',
        'MTL': 'data_vedio_dict/train_dict_MTL.json'
    }

    train_json_path = task_json[task]
    train_json_data = utils.get_json_data(train_json_path)
    train_data = get_train_data(train_json_data, args.min_time_step, args.max_time_step, args.forward_time_step)

    valid_json = 'data_vedio_dict/valid_dict.json'
    valid_json_data = utils.get_json_data(valid_json)
    valid_data = get_valid_data(valid_json_data)

    test_json = 'data_vedio_dict/test.json'
    test_json_data = utils.get_json_data(test_json)
    test_data = get_valid_data(test_json_data)

    train_loader = DataLoader(MyDataset(train_data, args.train_set_path, train_transformes), batch_size=1, shuffle=True)
    valid_loader = DataLoader(MyDataset(valid_data, args.valid_set_path, valid_transformes), batch_size=1)
    test_loader = DataLoader(MytestDataset(test_data, args.test_set_path, valid_transformes), batch_size=1)
    return train_loader, valid_loader, test_loader
