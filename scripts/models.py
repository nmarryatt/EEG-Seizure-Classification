import torch
import torch.nn as nn
import torch.nn.functional as F

## Simple Multi-Layer Perceptron: Baseline Model
class simple_EEG_MLP(nn.Module):
    def __init__(self, input_length=4097):
        super(simple_EEG_MLP, self).__init__()
        self.fc1 = nn.Linear(input_length, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # binary classification
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class simple_EEG_CNN(nn.Module):
    def __init__(self, input_length=4097):
        super(simple_EEG_CNN, self).__init__()

        # 1D Convolutional layers
        # Input  (32, 1, 4097)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3) 
        self.bn1 = nn.BatchNorm1d(16) # BatchNorm after conv1
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # after conv1 (32, 16, 4097), after pool1 (32, 16, 2048)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32) # BatchNorm after conv2
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # after conv 2  (32, 32, 2048), after pool2 (32, 32, 1024)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64) # BatchNorm after conv3
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2)
        # after conv 3 (32, 64, 1024), after pool3 (32, 64, 511))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * 511, 128)
        self.fc2 = nn.Linear(128, 2)  # binary classification output

    def forward(self, x):
        # Conv1 + BN + Relu + Pool
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv2 + BN + Relu + Pool
        x = self.conv2(x) 
        x = self.bn2(x) 
        x = F.relu(x) 
        x = self.pool2(x)

        # Conv3 + BN + ReLU + Pool 
        x = self.conv3(x) 
        x = self.bn3(x)
        x = F.relu(x) 
        x = self.pool3(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
