from cProfile import label
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision.datasets import VOCSegmentation, VOCDetection
import torch.utils.data
from torchvision import transforms
import numpy as np

import collections
import os
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
from PIL import Image
from typing import Any, Callable, Dict, Optional, Tuple, List

VOC_CLASSES_ID = {
    "background": 0,
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20,
}

VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
]

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]

class VocDetectionData(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        image_set="trainval",
        size=800,
    ):
        print("Initializing dataset")
        self.loader = VOCDetection(root=root, year="2012", image_set=image_set, transform=transforms.ToTensor())
        self.size = size
        self.resize = transforms.Resize((size, size))
        self.flip = transforms.RandomHorizontalFlip(p=1)
    
    def __getitem__(self, index):
        data = self.loader[index]
        image = data[0]
        c, h, w = image.size()
        image = self.resize(image)

        annotation = data[1]["annotation"]
        fname = annotation["filename"]
        objs = annotation["object"]
        
        labels = []
        bboxes = []
        for o in objs:
            labels.append(VOC_CLASSES_ID[o["name"]])
            xmin = int(o["bndbox"]["xmin"]) / w
            ymin = int(o["bndbox"]["ymin"]) / h
            xmax = int(o["bndbox"]["xmax"]) / w
            ymax = int(o["bndbox"]["ymax"]) / h
            bboxes.append(torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32))
        
        labels = torch.tensor(labels)
        bboxes = torch.stack(bboxes)
        
        if torch.rand(1) < 0.5:
            image = self.flip(image)
            xmin = 1 - bboxes[:, 0]
            xmax = 1 - bboxes[:, 2]
            bboxes[:, 0] = xmax
            bboxes[:, 2] = xmin

        return image, labels, bboxes, fname

    def __len__(self):
        return len(self.loader)


    
class VocSegmentationData(VOCSegmentation):
    _ANNOT_DIR = "Annotations"
    _ANNOT_FILE_EXT = ".xml"
    _TARGET_DIR = "SegmentationObject"

    @property
    def annotations(self) -> List[str]:
        return self.annot

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """

        base_dir = os.path.join("VOCdevkit", "VOC2012")
        voc_root = os.path.join(self.root, base_dir)
        splits_dir = os.path.join(voc_root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, self.image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        annot_dir = os.path.join(voc_root, self._ANNOT_DIR)
        self.annot = [os.path.join(annot_dir, x + self._ANNOT_FILE_EXT) for x in file_names]

        assert len(self.images) == len(self.annot)


        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])
        annotation = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        mask = np.array(mask)
        obj_ids = np.unique(mask)[1: -1]
        masks = mask == obj_ids[:, None, None]

        objs = annotation["annotation"]["object"]
        
        boxes = []
        labels = []
        for o in objs:
            labels.append(VOC_CLASSES_ID[o["name"]])
            xmin = int(o["bndbox"]["xmin"])
            ymin = int(o["bndbox"]["ymin"])
            xmax = int(o["bndbox"]["xmax"])
            ymax = int(o["bndbox"]["ymax"])
            boxes.append(torch.FloatTensor([xmin, ymin, xmax, ymax]))
        
        # # get bounding box coordinates for each mask
        # num_objs = len(obj_ids)
        # boxes = []
        # for i in range(num_objs):
        #     pos = np.where(masks[i])
        #     xmin = np.min(pos[1])
        #     xmax = np.max(pos[1])
        #     ymin = np.min(pos[0])
        #     ymax = np.max(pos[0])
        #     boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        # boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = torch.stack(boxes)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.tensor(labels, dtype=torch.int64)
        image_id = torch.tensor(index, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros(labels.shape[0], dtype=torch.uint8)


        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "masks": masks,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VocSegmentationData.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict