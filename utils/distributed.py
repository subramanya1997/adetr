import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def init_distributed_mode(args):
    """
    Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    """
    if args.device == "cpu":
        return 0
    
    env_dict = {
        key: os.environ[key]
        for key in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
    }
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device_id = rank % args.gpu_count
    seed = args.seed + rank
    print(f"==> Rank {rank} is using GPU {device_id}")
    print(f"==> Seed is {seed}")

    return device_id, seed, rank

def reduce_dict(input_dict, average=True):
    world_size = dist.get_world_size()
    with torch.no_grad():
        names = []
        values = []
        for k, v in input_dict.items():
            names.append(k)
            values.append(v)
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict