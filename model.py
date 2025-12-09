import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer 1: 1 input channel (grayscale), 16 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        # Max Pooling Layer 1: 2x2 kernel, stride 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Convolutional Layer 2: 16 input channels, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # Max Pooling Layer 2: 2x2 kernel, stride 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # The image starts at 28x28.
        # After pool1 (28/2), it becomes 14x14.
        # After pool2 (14/2), it becomes 7x7.
        # The number of output channels from conv2 is 32.
        # So, the input features to the fully connected layer is 32 * 7 * 7.
        self.fc1 = nn.Linear(32 * 7 * 7, 10) # 10 output classes for digits 0-9

    def forward(self, x):
        # Apply conv1, then ReLU activation, then pooling
        x = self.pool1(F.relu(self.conv1(x)))
        # Apply conv2, then ReLU activation, then pooling
        x = self.pool2(F.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 32 * 7 * 7)
        # Apply the fully connected layer
        x = self.fc1(x)
        return x
