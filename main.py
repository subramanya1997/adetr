# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import os
import sys
from time import time
import random
from copy import deepcopy
import numpy as np
from functools import partial
import wandb

import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from argsparser import parse_arguments
from models import build_model
from engine import train_one_epoch

from datasets import build_dataset
from utils.distributed import init_distributed_mode
from utils.misc import collate_fn

def init_wandb(run_name: str = "default"):
    return wandb.init(project="adetr", entity="subramaya1997", name=run_name)

def main(args):
    # initialize distributed mode
    print("==> Initializing distributed mode..")
    device_id, seed, rank = init_distributed_mode(args)
    logger = None
    if rank == 0 and args.log_to_wandb:
        logger = init_wandb(args.run_name)

    # fix the seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build the model
    print("==> Building model..")
    model, criterion, contrastive_criterion, weight_dict = build_model(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"==> Number of parameters: {n_parameters}")

    # set up the optimizer
    print("==> Setting up optimizer..")
    param_dicts = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if "backbone" not in n and "text_encoder" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad
            ],
            "lr": args.text_encoder_lr,
        }
    ]
    if args.optimizer == "sgd":
        optimizer = optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer in ["adam", "adamw"]:
        optimizer = optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    print("==> Moving model to DDP..")
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])
    model_without_ddp = model.module

    # build the dataset
    print("==> Preparing data..")
    if len(args.datasets) == 0:
        raise ValueError("No dataset specified")

    dataset_train = build_dataset(image_set="train", args=args)
    sampler_train = DistributedSampler(dataset_train)
    batch_sampler_train = BatchSampler(sampler_train, args.batch_size, drop_last=True)
    dataloader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=partial(collate_fn, False), num_workers=args.num_workers
    )

    dataset_val = build_dataset(image_set="val", args=args)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
    batch_sampler_val = BatchSampler(sampler_train, args.batch_size, drop_last=False)
    dataloader_val = DataLoader(
        dataset_val, batch_sampler=batch_sampler_val,
        collate_fn=partial(collate_fn, False), num_workers=args.num_workers
    )

    if args.load is not None:
        print(f"==> Loading checkpoint {args.load}..")
        checkpoint = torch.load(args.load, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model_ema"], strict=False)
        model_ema = deepcopy(model_without_ddp)

    if args.resume:
        args.start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer"])

    # start training
    print("==> Start training..")
    start_time = time()
    best_metric = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        print(f"==> Starting epoch {epoch}..")
        sampler_train.set_epoch(epoch)

        # train for one epoch
        train_stats = train_one_epoch(
            model=model,
            criterion=criterion,
            contrastive_criterion=contrastive_criterion,
            weight_dict=weight_dict,
            data_loader=dataloader_train,
            optimizer=optimizer,
            device_id=device_id,
            epoch=epoch,
            args=args,
            max_norm=args.clip_max_norm,
            model_ema=model_ema,
            logger=logger
        )
        break
        # evaluate on the validation set
        # if epoch % args.eval_skip == 0:
        #     pass

        print(f"==> Time for epoch {epoch}: {time() - start_time:.2f} s")

    print(f"==> Total training time: {time() - start_time:.2f} s")
    print("==> Done")



if __name__ == '__main__':
    args = parse_arguments()
    main(args)