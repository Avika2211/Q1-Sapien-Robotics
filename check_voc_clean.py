import os

# ===============================
# CONFIG
# ===============================
DATA_ROOT = "data/VOC_clean"
IMG_DIR = os.path.join(DATA_ROOT, "JPEGImages")
ANN_DIR = os.path.join(DATA_ROOT, "Annotations")
MAIN_DIR = os.path.join(DATA_ROOT, "ImageSets", "Main")

TRAIN_FILE = os.path.join(MAIN_DIR, "train.txt")
VAL_FILE   = os.path.join(MAIN_DIR, "val.txt")

# ===============================
# HELPER FUNCTION
# ===============================
def check_images_annotations():
    """Check if every image has a corresponding annotation"""
    missing_ann = []
    for img_file in os.listdir(IMG_DIR):
        if not img_file.lower().endswith(".jpg"):
            continue
        img_id = os.path.splitext(img_file)[0]
        ann_file = os.path.join(ANN_DIR, img_id + ".xml")
        if not os.path.exists(ann_file):
            missing_ann.append(img_file)
    if missing_ann:
        print(f"[ERROR] Missing annotations for {len(missing_ann)} images")
        for m in missing_ann:
            print(" -", m)
    else:
        print(f"[OK] All {len(os.listdir(IMG_DIR))} images have annotations.")

# ===============================
# HELPER FUNCTION
# ===============================
def check_splits():
    """Check that all images in train/val exist in JPEGImages"""
    for split_file, split_name in [(TRAIN_FILE, "train"), (VAL_FILE, "val")]:
        if not os.path.exists(split_file):
            print(f"[ERROR] {split_name}.txt missing")
            continue
        with open(split_file, "r") as f:
            ids = [line.strip() for line in f.readlines()]
        missing_imgs = []
        for img_id in ids:
            img_path = os.path.join(IMG_DIR, img_id + ".jpg")
            if not os.path.exists(img_path):
                missing_imgs.append(img_id)
        if missing_imgs:
            print(f"[ERROR] {len(missing_imgs)} missing images in {split_name}.txt")
        else:
            print(f"[OK] All {len(ids)} images in {split_name}.txt exist.")

# ===============================
# RUN CHECKS
# ===============================
if __name__ == "__main__":
    print("[INFO] Checking images and annotations...")
    check_images_annotations()
    print("[INFO] Checking train/val splits...")
    check_splits()
    print("[DONE] VOC_clean sanity check complete!")
