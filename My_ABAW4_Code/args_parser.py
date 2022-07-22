import argparse

def get_args_parser():
    parser = argparse.ArgumentParser()
    '''Base'''
    parser.add_argument('--devices', default='0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=7, type=int,
                        help='rand seed')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--train_epoch_nums', default=100, type=int)
    parser.add_argument('--loss_name', default='MTL', type=str,
                        choices=['crossentropy', 'MTL'])
    parser.add_argument('--lr', default=1e-3, type=float)
    

    '''EmotionNet'''
    parser.add_argument('--feature_extractor', default='resnet', type=str,
                        choices=['resnet', 'inception_v3'])
    parser.add_argument('--emotion_class_nums', default=8, type=int)
    parser.add_argument('--AU_nums', default=24, type=int)
    parser.add_argument('--hidden_AU_ratio', default=4.0, type=float)
    parser.add_argument('--AU_metric_dim', default=24, type=int)

    '''Transformer'''
    parser.add_argument('--all_drop', default=0.3, type=float)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--per_encoder_layer_nums', default=1, type=int)
    
    parser.add_argument('--qkv_bias', default=True)
    parser.add_argument('--mlp_bias', default=True)
    parser.add_argument('--feature_drop', default=0.)
    parser.add_argument('--attn_drop', default=0.)
    parser.add_argument('--mlp_drop', default=0.)

    '''Lr Scheduler'''
    parser.add_argument('--loss_mode', default='MTL', type=str,
                        choices=['AU', 'VA', 'EXPR', 'MTL'])
    parser.add_argument('--optimizer_name', default='adam', type=str,
                        choices=['adam'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--last_epoch', default=-1, type=int)
    

    '''Log'''
    parser.add_argument('--if_log', default=True, type=bool)
    parser.add_argument('--log_path', default='/log', type=str)
    parser.add_argument('--log_dir_name', default='test', type=str)

    '''Data'''
    parser.add_argument('--train_annotations', default='training_set_annotations.txt', type=str)
    parser.add_argument('--valid_annotations', default='validation_set_annotations.txt', type=str)
    
    parser.add_argument('--train_set_path', default='cropped_aligned', type=str)
    parser.add_argument('--valid_set_path', default='cropped_aligned', type=str)
    parser.add_argument('--test_set_path', default='cropped_aligned', type=str)
    
    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--batch_step', default=400, type=int)
    parser.add_argument('--min_time_step', default=10, type=int)
    parser.add_argument('--max_time_step', default=64, type=int)
    parser.add_argument('--forward_time_step', default=64, type=int, help='overlap_time_step = max_time_step - forward_time_step')

    return parser