# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
from .adetr import build

def build_model(args):
    return build(args)
