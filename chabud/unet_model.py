import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetChangeClassifier(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="timm-efficientnet-b0",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_channels,
            activation=None,
        )
        self.normalize = nn.BatchNorm2d(in_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        x = self.normalize(x)
        return self.model(x)
