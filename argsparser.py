# Copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Learning attributes of objects training and evaluation script')
    # config
    parser.add_argument('--config', type=str, default='./configs/config.yaml', help='dataset_config')
    # mode 
    parser.add_argument('--mode', type=str, default='train', help='train or test')

    try:
        args = parser.parse_args()
    except (IOError) as msg:
        parser.error(str(msg))

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        args.__dict__.update(config)

    print(args)
    
    return args