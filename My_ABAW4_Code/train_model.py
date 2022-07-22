import os
import sys
import utils
from args_parser import get_args_parser
from data_code import get_data, get_data_divide, get_data_for_task
from model_code import backbone_code, loss_code, optimizer_code, lr_scheduler_code
from process_code import train_epoch_code, test_epoch_code, inference_epoch_code
# CUDA_LAUNCH_BLOCKING=1
import torch
torch.autograd.set_detect_anomaly(True)

def train(args):
    utils.setup_seed(args.seed)

    if args.if_log:
        file_name = 'train_event.log'
        time_dir_name = utils.get_time_dir()
        log_dir_path = os.path.join(args.log_path, args.log_dir_name, time_dir_name)
        utils.create_dirs(log_dir_path)

        if hasattr(sys.stdout, 'reset_path'):
            sys.stdout.reset_path(os.path.join(log_dir_path, file_name))
        else:
            sys.stdout = utils.Logger(os.path.join(log_dir_path, file_name))

    task = args.loss_mode
    train_loader, valid_loader, test_loader = get_data_for_task.get_task_train_data(args, task)

    model = backbone_code.get_backbone(args)
    utils.set_parallel_devices(args.devices, model)
    model = model.cuda()
    model.set_gpu()

    loss = loss_code.get_loss(args)
    optimizer2 = None
    if args.feature_extractor == 'resnet':
        optimizer = torch.optim.Adam(model.stream.parameters(), lr=args.lr)
        optimizer2 = torch.optim.Adam(model.feature_extractor.parameters(), lr=args.lr)
    else:
        optimizer = optimizer_code.get_optimizer(args, model)
    sys.stdout.reset_path(os.path.join(log_dir_path, f'args.log'))
    print(args)
    sys.stdout.reset_path(os.path.join(log_dir_path, file_name), 'a')
    max_score = -1.
    for epoch in range(args.train_epoch_nums):
        train_epoch_code.train_one_epoch(epoch, model, optimizer, loss, iter(train_loader), args, optimizer2)
        score = test_epoch_code.test_one_epoch(model, iter(valid_loader), args)
        if score > max_score:
            max_score = score
            torch.save(model,os.path.join(log_dir_path, 'best_model.pth'))
            ans = inference_epoch_code.test_one_epoch(model, iter(test_loader), args)
            sys.stdout.reset_path(os.path.join(log_dir_path, f'ans_{score}.log'))
            print(ans)
            sys.stdout.reset_path(os.path.join(log_dir_path, file_name),'a')
    print(f'max_score = {max_score}')




if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    train(args)
