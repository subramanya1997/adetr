# copyright (c) Subramanya N. Licensed under the Apache License 2.0. All Rights Reserved
import os
import json
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import datasets.transforms as T


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
            for i, instance in enumerate(ann["instances"]):
                att_ids = []
                for att_name in instance["attributes"]:
                    self.att_img_map[self.att_id_map[att_name]].append(ann["image_id"])
                    att_ids.append(self.att_id_map[att_name])
                ann["instances"][i]["attributes"] = att_ids
            self.anns[ann["id"]] = ann
            self.img_ann_map[ann["image_id"]].append(ann["id"])
    
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
    
    def get_ann(self, ann_id):
        """
        Get annotation with the given id.
        args:
            ann_id: id of the annotation
        """
        return self.anns[ann_id]

    def get_img(self, img_id):
        """
        Get image with the given id.
        args:
            img_id: id of the image
        """
        return self.imgs[img_id]

    def __repr__(self):
        return f"VAW dataset with {len(self.imgs)} images, {len(self.atts)} attribute and {len(self.anns)} annotations"


class VAWDataset(Dataset):
    def __init__(self, img_dir, ann_path, transforms=None):
        self.img_dir = img_dir
        self.vaw = VAW(annotation_path=ann_path)
        self.ids = list(self.vaw.anns.keys())
        self.transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks=False)
    
    def __getitem__(self, idx):
        target = self.vaw.get_ann(ann_id=self.ids[idx])
        img_data = self.vaw.get_img(img_id=target["image_id"])
        img_path = os.path.join(self.img_dir, img_data['file_name'])
        img = Image.open(img_path).convert("RGB")
        img, target = self.prepare(img, target)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img_path, target

    def __len__(self):
        return len(self.ids)

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False, return_tokens=False, tokenizer=None):
        self.return_masks = return_masks
        self.return_tokens = return_tokens
        self.tokenizer = tokenizer

    def __call__(self, image, target):
        w, h = image.size
        idx = target["id"]
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        caption = target["caption"] if "caption" in target else None

        anno = target["instances"]
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [[att for att in obj["attributes"]] for obj in anno ]
        max_classes_length = max([len(c) for c in classes])
        classes = [c + [-100] * (max_classes_length - len(c)) for c in classes]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if caption is not None:
            target["caption"] = caption
        target["image_id"] = image_id
        target["id"] = idx

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        return image, target

def make_coco_transforms():
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    max_size = 1333
    return T.Compose([
            T.RandomResize([800], max_size=max_size),
            normalize,
        ]
    )

def build_vaw(image_set: str, args):
    vaw_path = args.vaw_dataset_path
    imgs_dir = args.vaw_imgs_dir
    assert os.path.exists(vaw_path), f"VAW dataset path {vaw_path} does not exist"
    dataset_path = os.path.join(vaw_path, f"final_{image_set}_data.json")
    assert os.path.exists(dataset_path), f"VAW dataset path {dataset_path} does not exist"
    dataset = VAWDataset(img_dir=imgs_dir, 
        ann_path=dataset_path, 
        transforms=make_coco_transforms()
    )
    for i in tqdm(range(len(dataset)), desc=f"{image_set}"):
        img_path, target = dataset[i]
    return dataset