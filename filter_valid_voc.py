import os
import shutil

VOC = "data/VOCdevkit/VOC2012"
IMG_DIR = os.path.join(VOC, "JPEGImages")
ANN_DIR = os.path.join(VOC, "Annotations")
OUT = "data/VOC_clean"

OUT_IMG = os.path.join(OUT, "JPEGImages")
OUT_ANN = os.path.join(OUT, "Annotations")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_ANN, exist_ok=True)

valid = 0

for xml in os.listdir(ANN_DIR):
    name = xml.replace(".xml", "")
    img = name + ".jpg"

    if os.path.exists(os.path.join(IMG_DIR, img)):
        shutil.copy(
            os.path.join(IMG_DIR, img),
            os.path.join(OUT_IMG, img)
        )
        shutil.copy(
            os.path.join(ANN_DIR, xml),
            os.path.join(OUT_ANN, xml)
        )
        valid += 1

print(f"[OK] Clean samples: {valid}")
print(f"Saved at: {OUT}")
