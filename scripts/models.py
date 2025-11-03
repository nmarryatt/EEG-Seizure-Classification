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