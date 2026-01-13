import os
import random

# ===============================
# CONFIG
# ===============================
DATA_ROOT = "data/VOC_clean"
IMG_DIR = os.path.join(DATA_ROOT, "JPEGImages")
ANN_DIR = os.path.join(DATA_ROOT, "Annotations")
OUT_DIR = os.path.join(DATA_ROOT, "ImageSets", "Main")

TRAIN_RATIO = 0.8
SEED = 42

# ===============================
# CREATE OUTPUT DIR
# ===============================
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# LOAD VALID FILES
# ===============================
image_ids = []

for file in os.listdir(IMG_DIR):
    if not file.lower().endswith(".jpg"):
        continue

    img_id = os.path.splitext(file)[0]
    ann_path = os.path.join(ANN_DIR, img_id + ".xml")

    if os.path.exists(ann_path):
        image_ids.append(img_id)

print(f"[INFO] Total clean samples: {len(image_ids)}")

# ===============================
# SHUFFLE + SPLIT
# ===============================
random.seed(SEED)
random.shuffle(image_ids)

split_idx = int(len(image_ids) * TRAIN_RATIO)
train_ids = image_ids[:split_idx]
val_ids = image_ids[split_idx:]

# ===============================
# WRITE FILES
# ===============================
train_file = os.path.join(OUT_DIR, "train.txt")
val_file = os.path.join(OUT_DIR, "val.txt")

with open(train_file, "w") as f:
    for img_id in train_ids:
        f.write(img_id + "\n")

with open(val_file, "w") as f:
    for img_id in val_ids:
        f.write(img_id + "\n")

# ===============================
# DONE
# ===============================
print(f"[OK] Train samples: {len(train_ids)}")
print(f"[OK] Val samples  : {len(val_ids)}")
print(f"[SAVED] {train_file}")
print(f"[SAVED] {val_file}")
