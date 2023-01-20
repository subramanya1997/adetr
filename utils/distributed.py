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
    print(f"==> Rank {rank} is using GPU {device_id}")

    return device_id