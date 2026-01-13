# train_yolo.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import VOCDataset, CLASSES

# -------------------------------
# 1. Configs
# -------------------------------
DATA_ROOT = "data/VOC_clean"
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-3
IMG_SIZE = 224  # small images for fast training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = len(CLASSES)
SAVE_PATH = "yolo.pth"

# -------------------------------
# 2. Dataset & Dataloader
# -------------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

train_dataset = VOCDataset(DATA_ROOT, split="train", transforms=transform)
val_dataset   = VOCDataset(DATA_ROOT, split="val", transforms=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

print(f"[INFO] Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

# -------------------------------
# 3. Mini YOLO-style model
# -------------------------------
class MiniYOLO(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes*5)  # 5 = [x, y, w, h, conf] per class
        )

    def forward(self, x):
        return self.head(self.backbone(x))

model = MiniYOLO(NUM_CLASSES).to(DEVICE)

# -------------------------------
# 4. Loss & optimizer
# -------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# 5. Training loop
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, targets in train_loader:
        imgs = torch.stack(imgs).to(DEVICE)        # [B,3,H,W]
        y_true = torch.stack(targets).to(DEVICE)  # [B, num_classes*5]

        optimizer.zero_grad()
        y_pred = model(imgs)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

# -------------------------------
# 6. Save model
# -------------------------------
torch.save(model.state_dict(), SAVE_PATH)
print(f"[DONE] Model saved at {SAVE_PATH}")
