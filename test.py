import torch
from model import FeatureExtractor

model = FeatureExtractor()

dummy = torch.randn(1, 3, 224, 224)
out = model(dummy)

print("Output shape:", out.shape)