import preprocessing as pp
from models import FeedforwardAminoToStructure, FeedforwardStructureToStructure, CNN2D, CNN1D
import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils.data as data

def train_feedforward_amino_to_structure_model(inputs, targets, learning_rate=0.001, num_epochs=100) -> FeedforwardAminoToStructure:
    """
    Trains a feedforward neural network model that maps amino acid sequences to secondary structure sequences.
    """
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
    """
    Trains a feedforward neural network model that maps secondary structure sequences to secondary structure sequences.
    """
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


def train_cnn2d_model(inputs, targets, learning_rate=0.001, num_epochs=100):
    """
    Trains a 2D convolutional neural network model that maps amino acid sequences to secondary structure sequences.
    """
    inputs = torch.Tensor(inputs).unsqueeze(1)
    targets = torch.Tensor(targets)
    num_classes = targets.shape[1]
    
    model = CNN2D(num_classes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)
    loader = data.DataLoader(data.TensorDataset(inputs, targets), batch_size=32)
    
    model.train()
    for i in range(num_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
        if  i + 1 == 1 or (i+1) % 10 == 0:
            print(f"Loss in epoch {i+1}: {loss.item()}")
    
    return model


def train_cnn1d_model(inputs, targets, learning_rate=0.001, num_epochs=100):
    """
    Trains a 1D convolutional neural network model that maps amino acid sequences to secondary structure sequences.
    """
    inputs = torch.Tensor(inputs).unsqueeze(1)
    targets = torch.Tensor(targets)
    num_classes = targets.shape[1]
    
    model = CNN1D(num_classes)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=learning_rate)
    loader = data.DataLoader(data.TensorDataset(inputs, targets), batch_size=32)
    
    model.train()
    for i in range(num_epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)
            loss.backward()
            optimizer.step()
        if  i + 1 == 1 or (i+1) % 10 == 0:
            print(f"Loss in epoch {i+1}: {loss.item()}")
    
    return model


def save_trained_model(model, save_file_path):
    """
    Save trained model to given file.
    """
    torch.save(model, save_file_path)


if __name__ == "__main__":
    train_data_file = "data/protein-secondary-structure.train"
    test_data_file = "data/protein-secondary-structure.test"   
    
    save_file_path_amino = "pretrained/feedforward_amino_to_structure.pt"  
    save_file_path_structure = "pretrained/feedforward_structure_to_structure.pt"
    save_file_path_cnn_2d = "pretrained/cnn_2d.pt"
    save_file_path_cnn_1d = "pretrained/cnn_1d.pt"
    
    save_model = True
    
    train_amino_to_structure = False
    train_structure_to_structure = False
    train_CNN_2D = False
    train_CNN_1D = True    
    
    num_epochs = 1000
    learning_rate = 0.0003
    
    X_train_a_to_s, y_train_a_to_s, X_test_a_to_s, y_test_a_to_s = pp.get_feedforward_amino_to_structure_data_sets(train_data_file, test_data_file)
    X_train_s_to_s, y_train_s_to_s, X_test_s_to_s, y_test_s_to_s = pp.get_feedforward_structure_to_structure_data_sets(train_data_file, test_data_file)
    X_train_CNN_2D, y_train_CNN_2D, X_train_CNN_2D, y_train_CNN_2D = pp.get_CNN_2D_data_set(train_data_file, test_data_file, window_size=13)
    
    
    if train_amino_to_structure : 
        learning_rate = 0.001
        amino_to_structure_model = train_feedforward_amino_to_structure_model(X_train_a_to_s, y_train_a_to_s, learning_rate, num_epochs)
        if save_model:
            save_trained_model(amino_to_structure_model, save_file_path_amino)
    if train_structure_to_structure: 
        learning_rate = 0.001
        structure_to_structure_model = train_feedforward_structure_to_structure_model(X_train_s_to_s, y_train_s_to_s, learning_rate, num_epochs)
        if save_model:
            save_trained_model(structure_to_structure_model, save_file_path_structure)
            
    if train_CNN_2D:
        learning_rate = 0.1
        cnn_2d_model = train_cnn2d_model(X_train_CNN_2D, y_train_CNN_2D, learning_rate, num_epochs)
        if save_model:
            save_trained_model(cnn_2d_model, save_file_path_cnn_2d)
            
    if train_CNN_1D:
        learning_rate = 0.1
        num_epochs = 500 # 1000
        cnn_1d_model = train_cnn1d_model(X_train_a_to_s, y_train_a_to_s, learning_rate, num_epochs)
        if save_model:
            save_trained_model(cnn_1d_model, save_file_path_cnn_1d)
  
    