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

