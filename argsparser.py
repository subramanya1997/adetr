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
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--load', type=str, default=None, help='path to the checkpoint to load')
    parser.add_argument('--save_dir', type=str, default='./save', help='path to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='path to save the logs')
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument('--log_to_wandb', action='store_true', help='log to wandb')

    # model
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"),
        help="Type of positional embedding to use on top of the image features",
    )
    ## backbone
    parser.add_argument("--backbone", default="timm_tf_efficientnet_b5_ns", type=str,
        help="Name of the convolutional backbone to use such as resnet50 resnet101 timm_tf_efficientnet_b3_ns",
    )
    ## transformer
    parser.add_argument("--dim_feedforward", default=2048, type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument("--enc_layers", default=6, type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument("--dec_layers", default=6, type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument("--nheads", default=8, type=int, 
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--dropout", default=0.1, type=float, 
        help="Dropout applied in the transformer"
    )
    parser.add_argument("--hidden_dim", default=256, 
        type=int, help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument("--num_queries", default=100, type=int, help="Number of query slots")
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--no_pass_pos_and_query", dest="pass_pos_and_query", action="store_false",
        help="Disables passing the positional encodings to each attention layers",
    )

    # segmentation
    parser.add_argument("--masks", action="store_true")

    # hyperparameters
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=5e-5, type=float)
    parser.add_argument("--text_encoder_lr", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--text_encoder_type", default="roberta-base",
        choices=("roberta-base", "distilroberta-base", "roberta-large"),
    )
    parser.add_argument("--freeze_text_encoder", action="store_true", 
        help="Whether to freeze the weights of the text encoder"
    )
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--eval_skip", default=5, type=int, 
        help='do evaluation every "eval_skip" frames'
    )
    parser.add_argument("--clip_max_norm", default=0.1, type=float, 
        help="gradient clipping max norm"
    )

    # loss
    parser.add_argument("--no_aux_loss", dest="aux_loss", action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )
    parser.add_argument("--contrastive_loss", action="store_true", 
        help="Whether to add contrastive loss"
        )
    parser.add_argument("--no_contrastive_align_loss", dest="contrastive_align_loss", 
        action="store_false", help="Whether to add contrastive alignment loss",
    )
    parser.add_argument("--contrastive_loss_hdim", type=int, default=64,
        help="Projection head output size before computing normalized temperature-scaled cross entropy loss",
    )

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