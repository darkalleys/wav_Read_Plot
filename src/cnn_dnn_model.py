"""
CNN-DNN model for audio spectrogram analysis with A100 optimizations.
Implements robust normalization layers and proper batch handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for spectrogram analysis."""
    
    def __init__(self, input_channels=2, base_channels=64):
        super(CNNFeatureExtractor, self).__init__()
        
        # Use GroupNorm instead of BatchNorm to handle batch_size=1
        # GroupNorm works with any batch size and doesn't require batch statistics
        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, base_channels)  # 8 groups for 64 channels
        
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(16, base_channels*2)  # 16 groups for 128 channels
        
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1)
        self.norm3 = nn.GroupNorm(32, base_channels*4)  # 32 groups for 256 channels
        
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1)
        self.norm4 = nn.GroupNorm(64, base_channels*8)  # 64 groups for 512 channels
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.25)
        
        # Global average pooling to handle variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.norm3(self.conv3(x))))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.norm4(self.conv4(x))))
        x = self.dropout(x)
        
        # Global average pooling to reduce to (batch_size, channels, 1, 1)
        x = self.global_pool(x)
        # Flatten to (batch_size, channels)
        x = x.view(x.size(0), -1)
        
        return x


class DNNClassifier(nn.Module):
    """DNN classifier with robust normalization."""
    
    def __init__(self, cnn_features=512, additional_features=128, hidden_size=1024, num_classes=10):
        super(DNNClassifier, self).__init__()
        
        self.input_size = cnn_features + additional_features
        
        # Use LayerNorm instead of BatchNorm for better stability with varying batch sizes
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.norm2 = nn.LayerNorm(hidden_size // 2)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.norm3 = nn.LayerNorm(hidden_size // 4)
        
        self.fc4 = nn.Linear(hidden_size // 4, num_classes)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.norm1(self.fc1(x))))
        x = self.dropout(F.relu(self.norm2(self.fc2(x))))
        x = self.dropout(F.relu(self.norm3(self.fc3(x))))
        x = self.fc4(x)
        return x


class CNNDNN(nn.Module):
    """Complete CNN-DNN model for audio spectrogram analysis."""
    
    def __init__(self, input_channels=2, base_channels=64, additional_features=128, 
                 hidden_size=1024, num_classes=10):
        super(CNNDNN, self).__init__()
        
        self.cnn = CNNFeatureExtractor(input_channels, base_channels)
        
        # Calculate CNN output features
        cnn_features = base_channels * 8  # 512 features from CNN
        
        self.dnn = DNNClassifier(cnn_features, additional_features, hidden_size, num_classes)
        
    def forward(self, cnn_input, additional_features=None):
        """
        Forward pass through CNN-DNN model.
        
        Args:
            cnn_input: Tensor of shape (batch_size, channels, height, width)
            additional_features: Optional tensor of shape (batch_size, additional_features)
        
        Returns:
            Tensor of shape (batch_size, num_classes)
        """
        # Extract CNN features
        cnn_features = self.cnn(cnn_input)
        
        # Combine with additional features if provided
        if additional_features is not None:
            combined_features = torch.cat([cnn_features, additional_features], dim=1)
        else:
            # Create zeros for additional features if not provided
            batch_size = cnn_features.size(0)
            additional_features = torch.zeros(batch_size, 128, device=cnn_features.device)
            combined_features = torch.cat([cnn_features, additional_features], dim=1)
        
        # DNN classification
        output = self.dnn(combined_features)
        
        return output

    def get_feature_sizes(self, input_shape=(2, 64, 64)):
        """Get the size of features at each layer for debugging."""
        with torch.no_grad():
            x = torch.randn(1, *input_shape)
            cnn_features = self.cnn(x)
            print(f"CNN output shape: {cnn_features.shape}")
            
            additional_features = torch.randn(1, 128)
            combined_features = torch.cat([cnn_features, additional_features], dim=1)
            print(f"Combined features shape: {combined_features.shape}")
            
            output = self.dnn(combined_features)
            print(f"Final output shape: {output.shape}")
            
        return cnn_features.shape[1], combined_features.shape[1], output.shape[1]


def create_model_medium():
    """Create a medium-sized CNN-DNN model as requested in the problem statement."""
    return CNNDNN(
        input_channels=2,
        base_channels=64,
        additional_features=128,
        hidden_size=1024,
        num_classes=10
    )


if __name__ == "__main__":
    # Test the model
    model = create_model_medium()
    print("Created CNN-DNN model with medium configuration")
    
    # Test with sample input
    cnn_input = torch.randn(1, 2, 64, 64)  # Single sample to test batch_size=1 handling
    additional_features = torch.randn(1, 128)
    
    model.eval()
    with torch.no_grad():
        output = model(cnn_input, additional_features)
        print(f"Model output shape: {output.shape}")
    
    # Get feature sizes for debugging
    model.get_feature_sizes()