# Copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved


def build_dataset(image_set: str, args):
    if args.vaw_dataset:
        from datasets.vaw import build_vaw
        return build_vaw(image_set=image_set, args=args)