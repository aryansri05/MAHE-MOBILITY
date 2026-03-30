import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, out_channels=64):
        super().__init__()
        # Load pretrained ResNet18
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Remove the final pooling and FC layers to keep spatial info
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        
        # Projection layer to match your 64-channel requirement
        self.project = nn.Conv2d(512, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x) # Output is roughly (B, 512, 7, 15) for 224x480 input
        return self.project(x)