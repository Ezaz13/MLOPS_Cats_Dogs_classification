import pytest
import torch
import torch.nn as nn
from src.model_building import train_model

class TestModelBuilding:
    def test_build_model_structure(self):
        """Test that the model is built with the correct head."""
        num_classes = 2
        model = train_model.build_model(num_classes)
        
        # Check if it's a ResNet
        assert hasattr(model, "fc")
        
        # Check if final layer has correct output features
        assert isinstance(model.fc, nn.Linear)
        assert model.fc.out_features == num_classes

    def test_model_forward_pass(self):
        """Test a dummy forward pass to ensure the model runs."""
        num_classes = 2
        model = train_model.build_model(num_classes)
        model.eval()
        
        # Create a dummy batch: (Batch Size, Channels, Height, Width)
        # ResNet expects 3 channels, 224x224 usually
        dummy_input = torch.randn(1, 3, 224, 224).to(train_model.DEVICE)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        # Output shape should be (1, num_classes)
        assert output.shape == (1, num_classes)
