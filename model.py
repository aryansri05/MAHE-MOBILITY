import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, out_channels=64):
        super(ResNetFeatureExtractor, self).__init__()
        
        # 1. Load the pre-trained ResNet-18 backbone
        # We use DEFAULT weights to get the best pre-trained ImageNet features
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Chop off the Classification Head
        # We delete the 'avgpool' and 'fc' (fully connected) layers at the end.
        # We stop at 'layer3' to keep the model lightweight and fast for the hackathon.
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1, # Outputs 64 channels
            resnet.layer2, # Outputs 128 channels
            resnet.layer3  # Outputs 256 channels
        )
        
        # 3. Compress the features (The Bottleneck)
        # We use a 1x1 Convolution to shrink the 256 channels down to a manageable size.
        # This saves massive amounts of compute when it gets passed to the View Transformer.
        self.compress = nn.Sequential(
            nn.Conv2d(256, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input 'x' is the image batch: shape (Batch, 3, Height, Width)
        
        # Pass through the chopped ResNet
        features = self.backbone(x) 
        
        # Compress the feature depth
        out = self.compress(features)
        
        # Output shape is smaller in spatial dimensions but deep in learned features
        return out

# --- TEST THE EXTRACTOR ---
if __name__ == "__main__":
    # Simulate a batch of 4 images from your Dataloader (from dataset.py)
    # 4 images, 3 color channels, 224 height, 480 width
    dummy_image_batch = torch.randn(4, 3, 224, 480) 
    
    # Initialize the model to output 64 feature channels
    extractor = ResNetFeatureExtractor(out_channels=64)
    
    # Pass the images through the network
    feature_map = extractor(dummy_image_batch)
    
    print("\n✅ --- Feature Extraction Complete --- ✅")
    print(f"Input Image Shape:  {dummy_image_batch.shape} (Batch, Channels, H, W)")
    print(f"Output Feature Map: {feature_map.shape} (Batch, Channels, Feature_H, Feature_W)")