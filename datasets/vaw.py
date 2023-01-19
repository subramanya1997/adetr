# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import os
import json
from collections import defaultdict
import torchvision

class VAW:
    def __init__(self, annotation_path=None):
        """
        class for reading and visualizing VAW dataset
        args:
            annotation_path: path to the json file containing annotations
        """
        self.anns = {}
        self.imgs = {}
        self.atts = {}
        self.att_id_map = {}
        self.img_ann_map = defaultdict(list)
        self.att_img_map = defaultdict(list)

        assert os.path.exists(annotation_path), f"annotation_path {annotation_path} does not exist"

        if annotation_path is not None:
            print("Loading annotations...")

            dataset = self._load_json(annotation_path)
            assert type(dataset) == dict, "annotation file format {} not supported".format(type(self.dataset))
            self._create_index(dataset=dataset)
    
    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _create_index(self, dataset=None):
        print("Creating index...")
        for att in dataset["categories"]:
            self.atts[att["id"]] = att
            self.att_id_map[att["name"]] = att["id"]
        for img in dataset["images"]:
            self.imgs[img["id"]] = img
        for ann in dataset["annotations"]:
            self.anns[ann["id"]] = ann
            self.img_ann_map[ann["image_id"]].append(ann["id"])
            for instance in ann["instances"]:
                for att_name in instance["attributes"]:
                    self.att_img_map[self.att_id_map[att_name]].append(ann["image_id"])
    
    def get_ann_ids(self, img_ids=None, att_ids=None):
        """
        Get annotation ids that satisfy given filter conditions.
        args:
            img_ids: (optional) list of image ids
            att_ids: (optional) list of attribute ids
        """
        if img_ids is not None:
            img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        if att_ids is not None:
            att_ids = att_ids if isinstance(att_ids, list) else [att_ids]
        
        anns = []
        if img_ids is not None:
            for img_id in img_ids:
                for ann_id in self.img_ann_map[img_id]:
                    anns.append(self.anns[ann_id])
        else:
            anns = list(self.anns.values())
        
        if att_ids is None:
            return [ann["id"] for ann in anns]
        att_ids = set(att_ids)

        ann_ids = [
            _ann["id"] for _ann in anns
            for instance in _ann["instances"]
            for att_name in instance["attributes"]
            if self.att_id_map[att_name] in att_ids
        ]
        return ann_ids

    def get_att_ids(self, img_ids=None):
        """
        Get attribute ids that satisfy given filter conditions.
        args:
            img_ids: (optional) list of image ids
        """
        if img_ids is not None:
            img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        
        if img_ids is None:
            return self.atts.keys()
        att_ids = []
        for img_id in img_ids:
            for ann_id in self.img_ann_map[img_id]:
                for instance in self.anns[ann_id]["instances"]:
                    for att_name in instance["attributes"]:
                        att_ids.append(self.att_id_map[att_name])
        
        return list(set(att_ids))

    def get_img_ids(self, att_ids=None):
        """
        Get image ids that satisfy given filter conditions.
        args:
            att_ids: (optional) list of attribute ids
        """
        if att_ids is not None:
            att_ids = att_ids if isinstance(att_ids, list) else [att_ids]

        if att_ids is None:
            return self.imgs.keys()
        img_ids = []
        for att_id in att_ids:
            img_ids.extend(self.att_img_map[att_id])

        return list(set(img_ids))

    def __repr__(self):
        return f"VAW dataset with {len(self.imgs)} images, {len(self.atts)} attribute and {len(self.anns)} annotations"


def build_vaw(image_set: str, args):
    vaw_path = args.vaw_dataset_path
    imgs_dir = args.vaw_imgs_dir
    assert os.path.exists(vaw_path), f"VAW dataset path {vaw_path} does not exist"
    dataset_path = os.path.join(vaw_path, f"final_{image_set}_data.json")
    assert os.path.exists(dataset_path), f"VAW dataset path {dataset_path} does not exist"
    dataset = VAW(annotation_path=dataset_path)
    return dataset