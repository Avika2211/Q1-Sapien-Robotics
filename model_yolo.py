import torch.nn as nn

class YOLOMini(nn.Module):
    def __init__(self, S=7, C=5):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 20 * 20, 512),
            nn.ReLU(),
            nn.Linear(512, S * S * (5 + C))
        )

        self.S, self.C = S, C

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.view(-1, self.S, self.S, 5 + self.C)
