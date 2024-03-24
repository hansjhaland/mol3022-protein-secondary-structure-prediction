import torch
import numpy as np
import preprocessing as pp
from one_hot_encodings import amino_acid_one_hot, secondary_structure_one_hot

def get_feedforward_amino_to_structure_predictions(model, inputs):
    inputs = torch.Tensor(inputs)
    inputs = inputs.unsqueeze(0)
    # print("SHAPE:",inputs.shape)
    
    model.eval()
    with torch.no_grad():
        predictions = []
        for input in inputs:
            prediction = model(input)
            predictions.append(prediction.numpy())
    return np.asarray(predictions)

def get_feedforward_structure_to_structure_predictions(model, inputs):
    inputs = torch.Tensor(inputs)
    inputs = inputs.unsqueeze(0)
    # print("SHAPE:",inputs.shape)
    
    model.eval()
    with torch.no_grad():
        predictions = []
        for input in inputs:
            prediction = model(input)
            predictions.append(prediction.numpy())
    return np.asarray(predictions).squeeze(0)

def get_structure_to_structure_input_from_amino_to_structure_output(amino_to_structure_model, secondary_structure_one_hot, amino_windows, window_size=13):
    structure_windows = []
    for amino_window in amino_windows:
        # amino_window = pp.get_sequence_windows(amino_sequence, window_size, amino_acid_one_hot)
        # amino_window = np.asarray(amino_window)
        structure_predictions = get_feedforward_amino_to_structure_predictions(amino_to_structure_model, amino_window)
        structure_window = pp.get_sequence_windows(structure_predictions.tolist(), window_size, secondary_structure_one_hot)
        structure_windows.append(structure_window)
    
    structure_to_structure_input = [window for sequence in structure_windows for window in sequence]
    return np.asarray(structure_to_structure_input)
            

def get_full_feedforward_predictions(amino_to_structure_model, structure_to_structure_model, amino_windows) -> list[list[float]]:
    structure_to_structure_inputs = get_structure_to_structure_input_from_amino_to_structure_output(amino_to_structure_model, secondary_structure_one_hot, amino_windows, window_size=13)
    feedforward_predictions = get_feedforward_structure_to_structure_predictions(structure_to_structure_model, structure_to_structure_inputs)
    return feedforward_predictions
    
def get_classification_from_probabilities(probabilities: np.ndarray[np.ndarray[float]]) -> tuple[list[list[int]], list[str], list[float]]:
    classification_index = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)
    index_to_secondary_strucure = {"0": "h", "1": "e", "2": "_"}
    
    one_hot_classifications = []
    symbol_classifictions = []
    for index in classification_index:
        secondary_structure = index_to_secondary_strucure[str(index)]
        one_hot_encoding = secondary_structure_one_hot[secondary_structure]
        symbol_classifictions.append(secondary_structure)
        one_hot_classifications.append(one_hot_encoding)
 
    return one_hot_classifications, symbol_classifictions, confidences
    

def load_trained_model(load_file_path):
    model = torch.load(load_file_path)
    model.eval()
    return model


if __name__ == "__main__":
    import training as train
    import evaluation as eval
    
    
    train_data_file = "data/protein-secondary-structure.train"
    test_data_file = "data/protein-secondary-structure.test"     
    
    load_file_path_amino = "pretrained/feedforward_amino_to_structure.pt"  
    load_file_path_structure = "pretrained/feedforward_structure_to_structure.pt"
    
    load_model = True
    
    X_train_a_to_s, y_train_a_to_s, X_test_a_to_s, y_test_a_to_s = pp.get_feedforward_amino_to_structure_data_sets(train_data_file, test_data_file)
    X_train_s_to_s, y_train_s_to_s, X_test_s_to_s, y_test_s_to_s = pp.get_feedforward_structure_to_structure_data_sets(train_data_file, test_data_file)
    
    if load_model:
        amino_to_structure_model = load_trained_model(load_file_path_amino)
        structure_to_structure_model = load_trained_model(load_file_path_structure)
    else:
        amino_to_structure_model = train.train_feedforward_amino_to_structure_model(X_train_a_to_s, y_train_a_to_s, num_epochs=1000)
        structure_to_structure_model = train.train_feedforward_structure_to_structure_model(X_train_s_to_s, y_train_s_to_s, num_epochs=1000)
        
    predicted_probabilites = get_full_feedforward_predictions(amino_to_structure_model, structure_to_structure_model, X_test_a_to_s)
    
    print(predicted_probabilites.shape)
    
    predicted_one_hots, predicted_symbold, confidences = get_classification_from_probabilities(predicted_probabilites)
    
    confusion_matrix = eval.get_confusion_matrix(predicted_one_hots, y_test_s_to_s)
    
    print(confusion_matrix)
    
    [print(f"Sensitivity: {sens}, Specificity {spec}") for (sens, spec) in eval.get_sensitivity_and_specificity_from_confusion_matrix(confusion_matrix)]
    