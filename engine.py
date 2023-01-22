# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import math
import sys
from typing import Dict, Iterable, Optional
import wandb

import torch
import torch.nn as nn
import torch.optim

from utils.misc import targets_to


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    contrastive_criterion: nn.Module,
    weight_dict: Dict[str, float],
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device_id,
    epoch: int,
    args,
    max_norm: float = 0,
    model_ema: Optional[nn.Module] = None,
    logger = None,
):
    model.train()
    if criterion is not None:
        criterion.train()
    if contrastive_criterion is not None:
        contrastive_criterion.train()
    
    if logger is not None:
        logger.log({
            "Train/epoch": epoch
        })

    for i, batch_dict in enumerate(data_loader):
        samples = batch_dict["samples"].to(device_id)
        targets = batch_dict["targets"]
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device_id)
        memory_cache = model(samples, captions)
        outputs = model(samples, captions, 
            encode_and_save=False, memory_cache=memory_cache)
        for k, v in outputs.items():
            if k in ["tokenized", "aux_outputs"]:
                continue
            print(k, v.shape)
        break

    return None