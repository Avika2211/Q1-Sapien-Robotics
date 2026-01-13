# dataset.py
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# VOC classes
CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

class VOCDataset(Dataset):
    def __init__(self, root, split="train", transforms=None):
        """
        root   : path to VOC_clean folder
        split  : 'train' or 'val'
        transforms: torchvision transforms to apply on images
        """
        self.root = root
        self.transforms = transforms

        self.img_dir = os.path.join(root, "JPEGImages")
        self.ann_dir = os.path.join(root, "Annotations")
        self.split_file = os.path.join(root, "ImageSets", "Main", f"{split}.txt")

        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"[ERROR] Split file not found: {self.split_file}")
        with open(self.split_file) as f:
            self.imgs = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_id = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_id + ".jpg")
        ann_path = os.path.join(self.ann_dir, img_id + ".xml")

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Parse XML
        tree = ET.parse(ann_path)
        root = tree.getroot()

        # YOLO-style targets: [num_classes*5]
        target = torch.zeros(len(CLASSES)*5)  # [x, y, w, h, conf] per class

        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in CLASS_TO_IDX:
                continue
            cls_idx = CLASS_TO_IDX[cls_name]
            bndbox = obj.find("bndbox")
            x1 = float(bndbox.find("xmin").text)
            y1 = float(bndbox.find("ymin").text)
            x2 = float(bndbox.find("xmax").text)
            y2 = float(bndbox.find("ymax").text)

            # Convert to YOLO-style normalized [x_center, y_center, w, h]
            x_c = ((x1 + x2)/2) / W
            y_c = ((y1 + y2)/2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            conf = 1.0

            target[cls_idx*5: cls_idx*5+5] = torch.tensor([x_c, y_c, w, h, conf])

        if self.transforms:
            img = self.transforms(img)

        return img, target
