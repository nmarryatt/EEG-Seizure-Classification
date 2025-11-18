import torch
import torch.nn as nn
import torch.nn.functional as F

## Simple Multi-Layer Perceptron: Baseline Model
class simple_EEG_MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline for EEG seizure detection
    - BatchNorm for training stability
    - Balanced architecture (128→128→64)
    - Dropout after both hidden layers
    
    Args:
        input_length: Length of EEG segment (default 4097)
    """
    def __init__(self, input_length=4097):
        super(simple_EEG_MLP, self).__init__()
        
        # Input layer
        self.fc1 = nn.Linear(input_length, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        
        # Hidden layer 1
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        
        # Hidden layer 2
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        # Output layer
        self.fc4 = nn.Linear(64, 2)  # Binary classification
        
    def forward(self, x):
        # Flatten input (handles both (B,C,L) and (B,L) inputs)
        x = x.view(x.size(0), -1)
        
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output
        x = self.fc4(x)
        
        return x
    

class simple_EEG_CNN(nn.Module):
    def __init__(self, input_length=4097):
        super(simple_EEG_CNN, self).__init__()

        # Conv1: Capture seizure spike patterns (70-200ms)
        # Kernel=25 → ~145ms at 173Hz
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=25, stride=1, padding=12) 
        self.bn1 = nn.BatchNorm1d(16) # BatchNorm after conv1
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(0.3)
        # After conv1+pool1: (batch, 16, 1024)

        # Conv2: Capture medium-term patterns (300-500ms)
        # Kernel=15 → ~350ms in the pooled space
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=15, stride=1, padding=7)
        self.bn2 = nn.BatchNorm1d(32) # BatchNorm after conv2
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(0.4)
        # After conv3+pool3: (batch, 32, 256)

        # Conv3: Capture long-term evolution patterns
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm1d(64) # BatchNorm after conv3
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        # after conv 3 (batch, 64, 64)
        
        # Dropout for regularization
        self.dropout3 = nn.Dropout(0.5)

        # global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(64, 64)  
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)  # NEW
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)  # NEW
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout3(x)  # NEW
        
        # Global pooling and classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x
