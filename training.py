import preprocessing as pp
from models import FeedforwardAminoToStructure, FeedforwardStructureToStructure
import torch.nn as nn
import torch.optim as optim
import torch

def train_feedforward_amino_to_structure_model(inputs, targets, learning_rate=0.001, num_epochs=100) -> FeedforwardAminoToStructure:
    inputs = torch.Tensor(inputs)
    targets = torch.Tensor(targets)
    
    input_size = inputs.shape[1]
    num_classes = targets.shape[1]
    
    model = FeedforwardAminoToStructure(input_size=input_size, num_classes=num_classes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    
    model.train()
    for i in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        if  i + 1 == 1 or (i+1) % 10 == 0:
            print(f"Loss in epoch {i+1}: {loss.item()}")
    
    return model
   
    
def train_feedforward_structure_to_structure_model(inputs, targets, learning_rate=0.001, num_epochs=100) -> FeedforwardStructureToStructure:
    inputs = torch.Tensor(inputs)
    targets = torch.Tensor(targets)
    
    input_size = inputs.shape[1]
    num_classes = targets.shape[1]
    
    model = FeedforwardStructureToStructure(input_size=input_size, num_classes=num_classes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
    
    model.train()
    for i in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()
        if  i + 1 == 1 or (i+1) % 10 == 0:
            print(f"Loss in epoch {i+1}: {loss.item()}")
    
    return model


def save_trained_model(model, save_file_path):
    torch.save(model, save_file_path)


if __name__ == "__main__":
    train_data_file = "data/protein-secondary-structure.train"
    test_data_file = "data/protein-secondary-structure.test"   
    
    save_file_path_amino = "pretrained/feedforward_amino_to_structure.pt"  
    save_file_path_structure = "pretrained/feedforward_structure_to_structure.pt"
    
    X_train_a_to_s, y_train_a_to_s, X_test_a_to_s, y_test_a_to_s = pp.get_feedforward_amino_to_structure_data_sets(train_data_file, test_data_file)
    X_train_s_to_s, y_train_s_to_s, X_test_s_to_s, y_test_s_to_s = pp.get_feedforward_structure_to_structure_data_sets(train_data_file, test_data_file)
    
    num_epochs = 1000
    learning_rate = 0.001
    
    amino_to_structure_model = train_feedforward_amino_to_structure_model(X_train_a_to_s, y_train_a_to_s, learning_rate, num_epochs)
    structure_to_structure_model = train_feedforward_structure_to_structure_model(X_train_s_to_s, y_train_s_to_s, learning_rate, num_epochs)
    
    save_trained_model(amino_to_structure_model, save_file_path_amino)
    save_trained_model(structure_to_structure_model, save_file_path_structure)
    
    print(amino_to_structure_model)
    print(structure_to_structure_model)
    