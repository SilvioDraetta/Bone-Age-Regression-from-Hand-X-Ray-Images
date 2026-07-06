
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),


            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),


            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.net(x)   # [B, 256, H/16, W/16]
    



class FiLM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.gamma = nn.Linear(1, channels)
        self.beta = nn.Linear(1, channels)

    def forward(self, x, male):
        """
        x: [B, C, H, W]
        male: [B, 1]
        """

        gamma = self.gamma(male).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(male).unsqueeze(-1).unsqueeze(-1)

        return x * (1 + gamma) + beta
    

class BoneAgeModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = CNNEncoder()

        self.film = FiLM(256)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, image, male=None):

        x = self.encoder(image)   # [B, 256, H, W]

        if male is not None:
            x = self.film(x, male)

        x = self.pool(x)          # [B, 256, 1, 1]
        x = torch.flatten(x, 1)   # [B, 256]

        x = self.regressor(x)

        return x.squeeze(1)