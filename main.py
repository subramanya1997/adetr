# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved

import os
import sys

from argsparser import parse_arguments

from datasets import build_dataset
from utils.distributed import init_distributed_mode

def main(args):
    # initialize distributed mode
    device_id = init_distributed_mode(args)
    print(device_id)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)