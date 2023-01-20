# Copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import os
import argparse
import yaml

import torch

from utils.misc import save_json

def parse_arguments():
    parser = argparse.ArgumentParser(description='Learning attributes of objects training and evaluation script')
    # config
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help='dataset_config')
    # run 
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--run_name', type=str, default='default', help='name of the run')
    parser.add_argument('--seed', type=int, default=48, help='random seed')
    parser.add_argument('--resume', action='store_false', help='resume training')
    parser.add_argument('--load', type=str, default=None, help='path to the checkpoint to load')
    parser.add_argument('--save_dir', type=str, default='./save', help='path to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='path to save the logs')

    try:
        args = parser.parse_args()
    except (IOError) as msg:
        parser.error(str(msg))

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.__dict__.update(config)

    # create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.save_dir, args.run_name)) and args.resume:
        os.system(f'rm -rf {os.path.join(args.save_dir, args.run_name)}')
    os.makedirs(os.path.join(args.save_dir, args.run_name), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.run_name, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.run_name, 'results'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.run_name, 'configs'), exist_ok=True)

    os.makedirs(os.path.join(args.log_dir, args.run_name), exist_ok=True)

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.log_dir = os.path.join(args.log_dir, args.run_name)
    args.save_dir = os.path.join(args.save_dir, args.run_name)
    args.checkpoints_dir = os.path.join(args.save_dir, 'checkpoints')
    args.results_dir = os.path.join(args.save_dir, 'results')
    args.configs_dir = os.path.join(args.save_dir, 'configs')

    # save config
    save_json(args.__dict__, os.path.join(args.configs_dir, 'run_config.json'))

    # update args
    args.device = torch.device(args.device)

    return args