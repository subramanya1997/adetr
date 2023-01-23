# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import math
import sys
from typing import Dict, Iterable, Optional
import wandb

import torch
import torch.nn as nn
import torch.optim

from utils.misc import targets_to
from utils.distributed import reduce_dict


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
        positive_map = batch_dict["positive_map"].to(device_id)
        positive_att = batch_dict["positive_att"].to(device_id)
        targets = batch_dict["targets"]
        captions = [t["caption"] for t in targets]

        targets = targets_to(targets, device_id)
        memory_cache = model(samples, captions)
        # print("memory_cache", memory_cache)
        outputs = model(samples, captions, 
            encode_and_save=False, memory_cache=memory_cache)
        # print("outputs", outputs)

        loss_dict = {}
        if criterion is not None:
            loss_dict.update(criterion(outputs, targets, positive_map, positive_att))

        if contrastive_criterion is not None:
            assert memory_cache is not None
            contrastive_loss = contrastive_criterion(memory_cache["text_pooled_op"], memory_cache["img_pooled_op"])
            loss_dict["contrastive_loss"] = contrastive_loss

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()

        ## adjust learning rate

        ## update model ema

        if logger is not None:
            for l, v in loss_dict_reduced_scaled.items():
                logger.log({
                    f"Train/{l}": v.item()
                })
            logger.log({
                f"Train/loss": loss_value,
                f"Train/lr": optimizer.param_groups[0]["lr"],
                f"Train/lr_backbone": optimizer.param_groups[1]["lr"],
                f"Train/lr_text_encoder": optimizer.param_groups[2]["lr"],
            })

        print(loss_value)

        break

    return None