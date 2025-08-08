import torch 
import torch.nn as nn

class DualSRNet(nn.Module):
    def __init__(self):
        super(DualSRNet, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x1, x2):
        f1 = self.branch(x1)
        f2 = self.branch(x2)
        combined = torch.cat([f1, f2], dim=1)
        return self.fuse(combined)