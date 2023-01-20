# Copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import json


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)