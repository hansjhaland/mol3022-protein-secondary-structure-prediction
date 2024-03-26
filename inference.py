import torch
import numpy as np
import preprocessing as pp
from one_hot_encodings import secondary_structure_one_hot
from scipy import stats as st

def get_feedforward_amino_to_structure_predictions(model, inputs):
    inputs = torch.Tensor(inputs)
    inputs = inputs.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        predictions = []
        for input in inputs:
            prediction = model(input)
            predictions.append(prediction.numpy())
    return np.asarray(predictions).squeeze()


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


def get_next_full_amino_sequence(amino_windows, amino_sequence_length):
    amino_sequence = []
    for i in range(amino_sequence_length):
        amino_sequence.append(amino_windows[i])
    remaining_windows = amino_windows[amino_sequence_length:]
    return amino_sequence, remaining_windows


def get_structure_to_structure_input_from_amino_to_structure_output(amino_to_structure_model, secondary_structure_one_hot, amino_windows, amino_sequence_lengths, window_size=13, window_type="concatenate"):
    structure_windows = []
    for sequence_length in amino_sequence_lengths:
        amino_sequence, amino_windows = get_next_full_amino_sequence(amino_windows, sequence_length)
    
        predicted_structure_sequence = get_feedforward_amino_to_structure_predictions(amino_to_structure_model, amino_sequence)
        structure_sequence_windows = pp.get_concat_sequence_windows(predicted_structure_sequence.tolist(), window_size, secondary_structure_one_hot)
        structure_windows = [*structure_windows, *structure_sequence_windows]

    return np.asarray(structure_windows)
            

def get_full_feedforward_predictions(amino_to_structure_model, structure_to_structure_model, amino_windows, amino_sequence_lengths, window_type="concatenate") -> list[list[float]]:
    structure_to_structure_inputs = get_structure_to_structure_input_from_amino_to_structure_output(amino_to_structure_model, secondary_structure_one_hot, amino_windows, amino_sequence_lengths, window_size=13, window_type=window_type)
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


def get_cnn_1d_predictions(cnn_1d_model, inputs):
    inputs = torch.Tensor(inputs)
    inputs = inputs.unsqueeze(1)
    
    cnn_1d_model.eval()
    with torch.no_grad():
        predictions = []
        for input in inputs:
            prediction = cnn_1d_model(input.unsqueeze(0))
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
    
    
def get_ensemble_predictions(feedforward_models, cnn_2d_model, cnn_1d_model, inputs_1d, inputs_2d, sequence_lengths, verbose=False):
    feedforward_predictions = get_full_feedforward_predictions(feedforward_models[0], feedforward_models[1], inputs_1d, sequence_lengths)
    cnn_2d_predictions = get_cnn_2d_predictions(cnn_2d_model, inputs_2d)
    cnn_1d_predictions = get_cnn_1d_predictions(cnn_1d_model, inputs_1d)
    
    feedforward_one_hot_classifications, feedforward_symbol_classifications, feedforward_confidences = \
        get_classification_from_probabilities(feedforward_predictions, model_type="ff")
    cnn_2d_one_hot_classifications, cnn_2d_symbol_classifications, cnn_2d_confidences = \
        get_classification_from_probabilities(cnn_2d_predictions, model_type="cnn")
    cnn_1d_one_hot_classifications, cnn_1d_symbol_classifications, cnn_1d_confidences = \
        get_classification_from_probabilities(cnn_1d_predictions, model_type="cnn")
        
    # print(type(feedforward_one_hot_classifications), type(feedforward_symbol_classifications), type(feedforward_confidences))
    # print(type(cnn_2d_one_hot_classifications), type(cnn_2d_symbol_classifications), type(cnn_2d_confidences))
    # print(type(cnn_1d_one_hot_classifications), type(cnn_1d_symbol_classifications), type(cnn_1d_confidences))
        
    # feedforward_one_hot_classifications, feedforward_symbol_classifications = np.asarray(feedforward_one_hot_classifications), np.asarray(feedforward_symbol_classifications)
    # cnn_2d_one_hot_classifications, cnn_2d_symbol_classifications = np.asarray(cnn_2d_one_hot_classifications), np.asarray(cnn_2d_symbol_classifications)
    # cnn_1d_one_hot_classifications, cnn_1d_symbol_classifications = np.asarray(cnn_1d_one_hot_classifications), np.asarray(cnn_1d_symbol_classifications)

    # h = 0
    # e = 1
    # c = 2

    # print(st.mode([e,e,h],[e,e,h],[e,e,h]).mode, st.mode([e,e,h]).count)
    # print(st.mode([e, h, c]))
    index_to_secondary_strucure = {"0": "h", "1": "e", "2": "_"}
    ensemble_symbols = []
    ensemble_one_hots = []
    for i in range(len(feedforward_one_hot_classifications)):
        
        ff, ff_conf = feedforward_one_hot_classifications[i], feedforward_confidences[i]
        cnn2d, cnn2d_conf = cnn_2d_one_hot_classifications[i], cnn_2d_confidences[i]
        cnn1d, cnn1d_conf = cnn_1d_one_hot_classifications[i], cnn_1d_confidences[i]
        
        ff_index = np.argmax(ff)
        cnn2d_index = np.argmax(cnn2d)
        cnn1d_index = np.argmax(cnn1d)
        
        ensemble_index = None
        prediction_indexes = [ff_index, cnn2d_index, cnn1d_index]
        index_mode = st.mode(prediction_indexes)
        if index_mode.count > 0:
            ensemble_index = index_mode.mode
        else:
            prediction_indexes = [ff_index, cnn2d_index, cnn1d_index]
            prediction_confidences = [ff_conf, cnn2d_conf, cnn1d_conf]
            most_confident = np.argmax(prediction_confidences)
            ensemble_index = prediction_indexes[most_confident]
        
        ensemble_one_hot = [0, 0, 0]
        ensemble_one_hot[ensemble_index] = 1
        ensemble_one_hots.append(ensemble_one_hot)
        ensemble_symbols.append(index_to_secondary_strucure[str(ensemble_index)])

    return ensemble_one_hots, ensemble_symbols 


def print_predicted_sequence():
    pass    


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
    load_file_cnn_1d = "pretrained/cnn_1d.pt"
    
    load_model = True
    
    # Feedfoward
    X_train_a_to_s, y_train_a_to_s, X_test_a_to_s, y_test_a_to_s, sequence_lengts_train, sequence_lengths_test = pp.get_feedforward_amino_to_structure_data_sets(train_data_file, test_data_file)
    X_train_s_to_s, y_train_s_to_s, X_test_s_to_s, y_test_s_to_s = pp.get_feedforward_structure_to_structure_data_sets(train_data_file, test_data_file)    
    
    # CNN
    X_train_CNN_2d, y_train_CNN_2d, X_test_CNN_2d, y_test_CNN_2d = pp.get_CNN_2D_data_set(train_data_file, test_data_file)
    
    
    if load_model:
        amino_to_structure_model = load_trained_model(load_file_path_amino)
        structure_to_structure_model = load_trained_model(load_file_path_structure)
        cnn_2d_model = load_trained_model(load_file_cnn_2d)
        cnn_1d_model = load_trained_model(load_file_cnn_1d)
    else:
        amino_to_structure_model = train.train_feedforward_amino_to_structure_model(X_train_a_to_s, y_train_a_to_s, num_epochs=1000)
        structure_to_structure_model = train.train_feedforward_structure_to_structure_model(X_train_s_to_s, y_train_s_to_s, num_epochs=1000)
        
    feedforward_predicted_probabilities = get_full_feedforward_predictions(amino_to_structure_model, structure_to_structure_model, X_test_a_to_s, sequence_lengths_test)
    cnn_predicted_probabilities = get_cnn_2d_predictions(cnn_2d_model, X_test_CNN_2d)
    cnn_1d_predicted_probabilities = get_cnn_1d_predictions(cnn_1d_model, X_test_a_to_s)
    
    
    feedforward_models = (amino_to_structure_model, structure_to_structure_model)
    one_hots, symbols = get_ensemble_predictions(feedforward_models, cnn_2d_model, cnn_1d_model, X_test_a_to_s, X_test_CNN_2d, sequence_lengths_test)
       
    print("Ensemble")
    confusion_matrix = eval.get_confusion_matrix(one_hots, y_test_s_to_s)
    [print(row) for row in confusion_matrix]
    [print(f"Sensitivity: {sens}, Specificity {spec}") for (sens, spec) in eval.get_sensitivity_and_specificity_from_confusion_matrix(confusion_matrix)]
    print()
    
    # print("Feedforward model")
    # feedforward_predicted_one_hots, feedforward_predicted_symbold, feedforward_confidences = get_classification_from_probabilities(feedforward_predicted_probabilities, model_type="ff")
    # feedforward_confusion_matrix = eval.get_confusion_matrix(feedforward_predicted_one_hots, y_test_s_to_s)
    # [print(row) for row in feedforward_confusion_matrix]
    # [print(f"Sensitivity: {sens}, Specificity {spec}") for (sens, spec) in eval.get_sensitivity_and_specificity_from_confusion_matrix(feedforward_confusion_matrix)]
    # print()
    
    # print("2D CNN")
    # cnn_predicted_one_hots, cnn_predicted_symbold, cnn_confidences = get_classification_from_probabilities(cnn_predicted_probabilities, model_type="cnn")
    # cnn_confusion_matrix = eval.get_confusion_matrix(cnn_predicted_one_hots, y_test_s_to_s)
    # [print(row) for row in cnn_confusion_matrix]
    # [print(f"Sensitivity: {sens}, Specificity {spec}") for (sens, spec) in eval.get_sensitivity_and_specificity_from_confusion_matrix(cnn_confusion_matrix)]
    # print()
    
    # print("1D CNN")
    # cnn_1d_predicted_one_hots, cnn_1d_predicted_symbold, cnn_1d_confidences = get_classification_from_probabilities(cnn_1d_predicted_probabilities, model_type="cnn")
    # cnn_1d_confusion_matrix = eval.get_confusion_matrix(cnn_1d_predicted_one_hots, y_test_s_to_s)
    # [print(row) for row in cnn_1d_confusion_matrix]
    # [print(f"Sensitivity: {sens}, Specificity {spec}") for (sens, spec) in eval.get_sensitivity_and_specificity_from_confusion_matrix(cnn_1d_confusion_matrix)]
        