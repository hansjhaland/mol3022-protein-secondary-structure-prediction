import preprocessing as pp
import encodings as one_hot
import numpy as np
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
    
def get_feedforward_amino_to_structure_predictions():
    pass


# NOTE: This is relevant when doing inference with both FF models
# def get_feedforward_structure_to_structure_model_inputs(amino_to_structure_model, 
#                                                         window_size,
#                                                         amino_acid_one_hot,
#                                                         secondary_structure_one_hot,
#                                                         train_data_file):
#     sequence_pairs = pp.load_sequences_from_file(train_data_file)
#     amino_sequence = [sequence[0] for sequence in sequence_pairs]
#     amino_sequence_one_hots = [pp.get_one_hot_encoding_of_sequence(sequence, amino_acid_one_hot) for sequence in amino_sequence]

#     structure_windows = []
#     for amino_sequence in amino_sequence_one_hots:
#         amino_window = pp.get_sequence_windows(amino_sequence, window_size, amino_acid_one_hot)
#         amino_window = np.asarray(amino_window)
#         structure_classification_probabilities = get_feedforward_amino_to_structure_predictions(amino_to_structure_model,
#                                                                                                 amino_window)


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


def load_trained_model(load_file_path):
    # Filename based on type
    model = torch.load(load_file_path)
    model.eval()
    return model


def save_trained_model(model, save_file_path):
    torch.save(model, save_file_path)

if __name__ == "__main__":
    train_data_file = "data/protein-secondary-structure.train"
    test_data_file = "data/protein-secondary-structure.test"     
    
    X_train_a_to_s, y_train_a_to_s, X_test_a_to_s, y_test_a_to_s = pp.get_feedforward_amino_to_structure_data_sets(train_data_file, test_data_file)
    X_train_s_to_s, y_train_s_to_s, X_test_s_to_s, y_test_s_to_s = pp.get_feedforward_structure_to_structure_data_sets(train_data_file, test_data_file)
    
    amino_to_structure_model = train_feedforward_amino_to_structure_model(X_train_a_to_s, y_train_a_to_s)
    structure_to_structure_model = train_feedforward_structure_to_structure_model(X_train_s_to_s, y_train_s_to_s)
    
    print(amino_to_structure_model)
    print(structure_to_structure_model)
    