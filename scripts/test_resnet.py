import sys
import os

# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from ResNet.ResNetUNet3D import ResNetUNet3D

def test_inference():
    model = ResNetUNet3D(in_channels=1, out_channels=1)
    model.eval()
    
    # Create random input tensor
    x = torch.randn(1, 1, 64, 128, 128)
    
    print(f"Input shape: {x.shape}")
    
    # Run inference
    with torch.no_grad():
        output = model(x)
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
    
    print(f"Output shape: {output.shape}")
    
    print("\nRaw output statistics:")
    print(f"Min value: {output.min().item():.3f}")
    print(f"Max value: {output.max().item():.3f}")
    print(f"Mean value: {output.mean().item():.3f}")
    
    print("\nProbability statistics (after sigmoid):")
    print(f"Min probability: {probs.min().item():.3f}")
    print(f"Max probability: {probs.max().item():.3f}")
    print(f"Mean probability: {probs.mean().item():.3f}")

if __name__ == "__main__":
    test_inference()