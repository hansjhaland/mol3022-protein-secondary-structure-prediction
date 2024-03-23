import torch
from torch import nn
import preprocessing as pp

torch.manual_seed(42)

class FeedforwardAminoToStructure(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__(self)
        self.input = nn.Linear(input_size, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, num_classes)
        
        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.2)
        self.do3 = nn.Dropout(0.2)
        
        self.softmax = nn.Softmax(num_classes)
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
    def __init__(self, input_size: int, num_classes: int):
        super().__init__(self)
        self.input = nn.Linear(input_size, 128)
        self.hidden1 = nn.Linear(128, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, num_classes)
        
        self.do1 = nn.Dropout(0.2)
        self.do2 = nn.Dropout(0.2)
        self.do3 = nn.Dropout(0.2)
        
        self.softmax = nn.Softmax(num_classes)
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