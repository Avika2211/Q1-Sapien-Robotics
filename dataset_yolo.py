import os, torch, xml.etree.ElementTree as ET
from PIL import Image
from torchvision import transforms

CLASSES = ["person", "car", "dog", "cat", "bicycle"]
C = len(CLASSES)

class VOCDatasetYOLO(torch.utils.data.Dataset):
    def __init__(self, root, split="train", S=7, img_size=320, max_samples=600):
        self.root = root
        self.S = S

        with open(f"{root}/ImageSets/Main/{split}.txt") as f:
            ids = [x.strip() for x in f.readlines()]

        self.ids = []
        for i in ids:
            if os.path.exists(f"{root}/Annotations/{i}.xml"):
                self.ids.append(i)

        self.ids = self.ids[:max_samples]

        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = self.tf(Image.open(
            f"{self.root}/JPEGImages/{img_id}.jpg"
        ).convert("RGB"))

        target = torch.zeros(self.S, self.S, 5 + C)

        tree = ET.parse(f"{self.root}/Annotations/{img_id}.xml")
        for obj in tree.findall("object"):
            cls = obj.find("name").text
            if cls not in CLASSES:
                continue

            box = obj.find("bndbox")
            xmin, ymin, xmax, ymax = map(
                int, [box.find(x).text for x in ["xmin","ymin","xmax","ymax"]]
            )

            x = (xmin + xmax) / 2 / img.shape[2]
            y = (ymin + ymax) / 2 / img.shape[1]
            w = (xmax - xmin) / img.shape[2]
            h = (ymax - ymin) / img.shape[1]

            i, j = int(self.S * y), int(self.S * x)
            target[i, j, :4] = torch.tensor([x, y, w, h])
            target[i, j, 4] = 1
            target[i, j, 5 + CLASSES.index(cls)] = 1

        return img, target
