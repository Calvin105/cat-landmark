import torch
import torch.nn as nn
import torchvision.models as models

class CatLandmarkModel(nn.Module):
    """
    Cat Landmark Regression Model using EfficientNet-B0 backbone.
    Predicts 48 facial landmark coordinates normalized in the [0, 1] range.
    """
    def __init__(self, pretrained=True):
        super(CatLandmarkModel, self).__init__()
        # Load the pretrained EfficientNet-B0 backbone
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Replace the ImageNet classification layer with the custom landmark regression head
        # Input features for EfficientNet-B0 classifier is 1280.
        # Output features is 96 (48 landmarks * 2 dimensions [x, y]).
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=96),
            nn.Sigmoid() # Restrict coordinate predictions to [0, 1]
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == '__main__':
    # Simple validation code
    print("Initializing model (without pre-trained weights for quick local check)...")
    model = CatLandmarkModel(pretrained=False)
    
    # Generate dummy batch: shape [batch_size, channels, height, width]
    # Input dimensions for EfficientNet are typically 224x224
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print(f"Running dummy forward pass with input shape: {dummy_input.shape}")
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 96), f"Expected shape (2, 96), got {output.shape}"
    assert (output >= 0.0).all() and (output <= 1.0).all(), "Outputs must be normalized in [0, 1]"
    
    print("Model initialized and verified successfully!")
