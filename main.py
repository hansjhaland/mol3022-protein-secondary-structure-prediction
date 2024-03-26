# Overall procedure (Based on slides page 37 and 38):
# - Read train and test data from files
# - Preprocessing of data to get sequences of one-hot encoded amino acids and secondary structures
# - Input data is organized into windows with additional padding one-hot encodings
# - Sequence-to-structure: Train neural network that takes one window as input and predict the probability of each secondary structure
# - Build a sequence of secondary structures by taking the highest probability secondary structure for each window
# - Create windows of the predicted secondary structure sequence
# - Structure-to-structure: Train second neural network that takes one window of the predicted secondary structure sequence as input 
#   and predict the probability of each secondary structure
# - Build a sequence of secondary structures by taking the highest probability secondary structure for each window
# - The resulting sequence is the output of the system

# import tensorflow as tf
# print("Tensorflow version:", tf.__version__)

# from preprocessing import get_data_sets_for_supervised_learning
# from one_hot_encodings import amino_acid_one_hot, secondary_structure_one_hot
# from preprocessing import get_one_hot_encoding

# pretrained_seq_model = tf.keras.models.load_model("models/seq_to_struct_model")
# pretrained_struct_model = tf.keras.models.load_model("models/struct_to_struct_model")


# # Convert probabilities to one-hot encodings
# y_predictions_index = tf.argmax(y_probabilities, axis=1)

# index_to_secondary_strucure = {"0": "h", "1": "e", "2": "_"}

# y_pred = []
# for index in y_predictions_index.numpy():
#     secondary_structure = index_to_secondary_strucure[str(index)]
#     one_hot_encoding = secondary_structure_one_hot[secondary_structure]
#     y_pred.append(one_hot_encoding)
    
# print(y_probabilities[:5])
# print(y_pred[:5])
# print(len(y_pred))

import inference as infer
import preprocessing as pp
from one_hot_encodings import amino_acid_one_hot
import numpy as np

def main():
    input_sequence: str = input("Enter sequence of amino acids: ").upper()
    
    print("Predicting secondary structure sequence...\n")
    
    sequence_length = len(input_sequence)
    
    one_hot_sequence: list[list[int]] = pp.get_one_hot_encoding_of_sequence(input_sequence, amino_acid_one_hot)
    
    window_size = 13
    
    sequence_windows_1d = pp.get_concat_sequence_windows(one_hot_sequence, window_size, amino_acid_one_hot)
    model_input_1d = sequence_windows_1d
    
    sequence_windows_2d = pp.get_sequential_sequence_windows(one_hot_sequence, window_size, amino_acid_one_hot)
    model_input_2d = [np.transpose(window) for window in sequence_windows_2d] 
    
    file_path_amino_to_structure = "pretrained/feedforward_amino_to_structure.pt" 
    file_path_structure_to_structure = "pretrained/feedforward_structure_to_structure.pt"
    file_path_2d_cnn = "pretrained/cnn_2d.pt"
    file_path_1d_cnn = "pretrained/cnn_1d.pt"
    
    two_stage_feedforward_model = (infer.load_trained_model(file_path_amino_to_structure), 
                                   infer.load_trained_model(file_path_structure_to_structure))
    cnn_2d_model = infer.load_trained_model(file_path_2d_cnn)
    cnn_1d_model = infer.load_trained_model(file_path_1d_cnn)
    
    _, symbols = infer.get_ensemble_predictions(two_stage_feedforward_model,
                                                       cnn_2d_model,
                                                       cnn_1d_model, 
                                                       model_input_1d,
                                                       model_input_2d,
                                                       [sequence_length])
    
    predicted_secondary_structure_sequence = "".join(symbols)
    
    print("Predicted secondary structure sequence is:")
    print(predicted_secondary_structure_sequence + "\n")
    
    compare: str = input("Compare result with true secondary structure sequence? (y/n): ").lower()
    
    if compare == "y":
        true_secondary_structure_sequence = input("Enter true sequence: ").lower()
        
        compare_string = []
        miss_count = 0
        for pred, true in zip(predicted_secondary_structure_sequence, 
                              true_secondary_structure_sequence):
            
            if pred.lower() == true.lower():
                compare_string.append(" ")
                continue
                
            compare_string.append("X")
            miss_count += 1
            
        compare_string = "".join(compare_string)
        
        print(f"Missed {miss_count} out of {sequence_length} symbols.")
        print("Showing sequences in order: misses, true, predicted. Position of misses marked by 'X':")
        print(compare_string)
        print()
        print(true_secondary_structure_sequence)  
        print()
        print(predicted_secondary_structure_sequence)
        print()
            

if __name__ == "__main__":
    main()

