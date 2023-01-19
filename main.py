# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved

import os
import sys

from argsparser import parse_arguments

from datasets import build_dataset

if __name__ == '__main__':
    args = parse_arguments()
    # t_dataset = build_dataset(image_set="train", args=args)
    v_dataset = build_dataset(image_set="val", args=args)
    print('---')