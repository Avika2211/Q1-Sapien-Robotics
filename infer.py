# infer_yolo.py
import os
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from dataset import CLASSES
from train_yolo import MiniYOLO  # mini model definition

# -------------------------------
# 1. Configs
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
MODEL_PATH = "yolo.pth"
INPUT_FOLDER = "data/VOC_clean/JPEGImages"
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# 2. Load model
# -------------------------------
NUM_CLASSES = len(CLASSES)
model = MiniYOLO(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------
# 3. Transforms
# -------------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

# -------------------------------
# 4. Inference loop
# -------------------------------
images_for_gif = []
for img_file in os.listdir(INPUT_FOLDER):
    if not img_file.lower().endswith((".jpg", ".png")):
        continue
    img_path = os.path.join(INPUT_FOLDER, img_file)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)  # [1, num_classes*5]
    pred = pred.squeeze(0).cpu()

    draw = ImageDraw.Draw(img)
    W, H = img.size

    # Simple YOLO-style decode (dummy, for mini YOLO demo)
    for c in range(NUM_CLASSES):
        x, y, w, h, conf = pred[c*5:c*5+5]
        if conf > 0.5:  # threshold
            # Convert normalized coords to image coords
            x0 = (x - w/2) * W
            y0 = (y - h/2) * H
            x1 = (x + w/2) * W
            y1 = (y + h/2) * H
            draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
            draw.text((x0, y0-10), f"{CLASSES[c]}:{conf:.2f}", fill="red")

    out_path = os.path.join(OUTPUT_FOLDER, img_file)
    img.save(out_path)
    images_for_gif.append(img)

print(f"[INFO] Inference done. Outputs saved in {OUTPUT_FOLDER}")

# -------------------------------
# 5. Create GIF
# -------------------------------
gif_path = os.path.join(OUTPUT_FOLDER, "detections.gif")
images_for_gif[0].save(
    gif_path,
    save_all=True,
    append_images=images_for_gif[1:],
    duration=400,
    loop=0
)
print(f"[DONE] GIF saved at {gif_path}")
