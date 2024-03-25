import torch
import numpy as np
import preprocessing as pp
from one_hot_encodings import secondary_structure_one_hot

def get_feedforward_amino_to_structure_predictions(model, inputs):
    inputs = torch.Tensor(inputs)
    inputs = inputs.unsqueeze(0)

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
        structure_predictions = get_feedforward_amino_to_structure_predictions(amino_to_structure_model, amino_window)
        structure_window = pp.get_sequence_windows(structure_predictions.tolist(), window_size, secondary_structure_one_hot)
        structure_windows.append(structure_window)
    
    structure_to_structure_input = [window for sequence in structure_windows for window in sequence]
    return np.asarray(structure_to_structure_input)
            

def get_full_feedforward_predictions(amino_to_structure_model, structure_to_structure_model, amino_windows) -> list[list[float]]:
    structure_to_structure_inputs = get_structure_to_structure_input_from_amino_to_structure_output(amino_to_structure_model, secondary_structure_one_hot, amino_windows, window_size=13)
    feedforward_predictions = get_feedforward_structure_to_structure_predictions(structure_to_structure_model, structure_to_structure_inputs)
    return feedforward_predictions
    
    
def get_cnn_2d_predictions(cnn_2d_model, inputs):
    inputs = torch.Tensor(inputs)
    inputs = inputs.unsqueeze(1)
    
    cnn_2d_model.eval()
    with torch.no_grad():
        predictions = []
        for input in inputs:
            prediction = cnn_2d_model(input.unsqueeze(0))
            predictions.append(prediction.numpy())
    return np.asarray(predictions)
    
    
def get_classification_from_probabilities(probabilities: np.ndarray[np.ndarray[float]], model_type="ff") -> tuple[list[list[int]], list[str], list[float]]:
    if model_type == "cnn":
        classification_index = np.argmax(probabilities, axis=2).squeeze()
        confidences = np.max(probabilities, axis=2).squeeze()
    else:
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
    load_file_cnn_2d = "pretrained/cnn_2d.pt"
    
    load_model = True
    
    # Feedfoward
    X_train_a_to_s, y_train_a_to_s, X_test_a_to_s, y_test_a_to_s = pp.get_feedforward_amino_to_structure_data_sets(train_data_file, test_data_file)
    X_train_s_to_s, y_train_s_to_s, X_test_s_to_s, y_test_s_to_s = pp.get_feedforward_structure_to_structure_data_sets(train_data_file, test_data_file)
    
    # CNN
    X_train_CNN_2d, y_train_CNN_2d, X_test_CNN_2d, y_test_CNN_2d = pp.get_CNN_2D_data_set(train_data_file, test_data_file)
    
    if load_model:
        amino_to_structure_model = load_trained_model(load_file_path_amino)
        structure_to_structure_model = load_trained_model(load_file_path_structure)
        cnn_2d_model = load_trained_model(load_file_cnn_2d)
    else:
        amino_to_structure_model = train.train_feedforward_amino_to_structure_model(X_train_a_to_s, y_train_a_to_s, num_epochs=1000)
        structure_to_structure_model = train.train_feedforward_structure_to_structure_model(X_train_s_to_s, y_train_s_to_s, num_epochs=1000)
        
    feedforward_predicted_probabilities = get_full_feedforward_predictions(amino_to_structure_model, structure_to_structure_model, X_test_a_to_s)
    cnn_predicted_probabilities = get_cnn_2d_predictions(cnn_2d_model, X_test_CNN_2d)
    
    feedforward_predicted_one_hots, feedforward_predicted_symbold, feedforward_confidences = get_classification_from_probabilities(feedforward_predicted_probabilities, model_type="ff")
    feedforward_confusion_matrix = eval.get_confusion_matrix(feedforward_predicted_one_hots, y_test_s_to_s)
    [print(row) for row in feedforward_confusion_matrix]
    [print(f"Sensitivity: {sens}, Specificity {spec}") for (sens, spec) in eval.get_sensitivity_and_specificity_from_confusion_matrix(feedforward_confusion_matrix)]
    
    cnn_predicted_one_hots, cnn_predicted_symbold, cnn_confidences = get_classification_from_probabilities(cnn_predicted_probabilities, model_type="cnn")
    cnn_confusion_matrix = eval.get_confusion_matrix(cnn_predicted_one_hots, y_test_s_to_s)
    [print(row) for row in cnn_confusion_matrix]
    [print(f"Sensitivity: {sens}, Specificity {spec}") for (sens, spec) in eval.get_sensitivity_and_specificity_from_confusion_matrix(cnn_confusion_matrix)]
    