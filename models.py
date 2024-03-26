import torch
from torch import nn

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)

class FeedforwardAminoToStructure(nn.Module):
    """
    Input: Window of one-encoded protein subsequence.
    Output: [p_helix, p_sheet, p_coil]
     
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input = nn.Linear(input_size, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, num_classes)
        
        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.2)
        self.do3 = nn.Dropout(0.2)
        
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.do1(x)
        x = self.relu(self.hidden1(x))
        x = self.do2(x)
        x = self.relu(self.hidden2(x))
        x = self.do3(x)
        x = self.relu(self.output(x))
        x = self.softmax(x)
        return x


class FeedforwardStructureToStructure(nn.Module):
    """
    Input: Window of secondary structure classification probabilities.
    Output: [p_helix, p_sheet, p_coil]
    
    """
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input = nn.Linear(input_size, 32)
        self.hidden1 = nn.Linear(32, 32)
        self.hidden2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, num_classes)
        
        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.2)
        self.do3 = nn.Dropout(0.2)
        
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.input(x))
        x = self.do1(x)
        x = self.relu(self.hidden1(x))
        x = self.do2(x)
        x = self.relu(self.hidden2(x))
        x = self.do3(x)
        x = self.relu(self.output(x))
        x = self.softmax(x)
        return x
    
    
class CNN2D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.conv2d = nn.Conv2d(1, 16, 5, padding="same")
        self.avg_pool = nn.AvgPool2d(4,4)
        
        self.hidden1 = nn.Linear(240, 240)
        
        self.output = nn.Linear(240, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        x = self.conv2d(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.hidden1(x))
        x = self.output(x)
        x = self.softmax(x)
        # print(x)
        return x
    
    
class CNN1D(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1d = nn.Conv1d(1, 1, 7, padding="same")
        # self.conv1d2 = nn.Conv1d(16, 1, 13, padding="same")
        self.avg_pool = nn.AvgPool1d(5,5)
    
        self.hidden1 = nn.Linear(52, 32)
        # self.hidden2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, num_classes)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1d(x))
        x = self.avg_pool(x)
        # x = self.relu(self.conv1d2(x))
        # x = self.avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.hidden1(x))
        # x = self.relu(self.hidden2(x))
        x = self.output(x)
        x = self.softmax(x)
        return x
    