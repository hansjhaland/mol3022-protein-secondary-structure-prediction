# TODO: Consider creating a file similar to main where only the 2D cnn model is used.
import inference as infer
import preprocessing as pp
from one_hot_encodings import amino_acid_one_hot
import numpy as np

def main():
    
    while True:
        input_sequence: str = input("Enter sequence of amino acids: ").upper()
        
        print("Predicting secondary structure sequence...\n")
        
        sequence_length = len(input_sequence)
        
        one_hot_sequence: list[list[int]] = pp.get_one_hot_encoding_of_sequence(input_sequence, amino_acid_one_hot)
        
        window_size = 13
        
        sequence_windows_2d = pp.get_sequential_sequence_windows(one_hot_sequence, window_size, amino_acid_one_hot)
        model_input_2d = [np.transpose(window) for window in sequence_windows_2d] 
        
        file_path_2d_cnn = "pretrained/cnn_2d.pt"

        cnn_2d_model = infer.load_trained_model(file_path_2d_cnn)

        predictions = infer.get_cnn_2d_predictions(cnn_2d_model, model_input_2d)
        
        _, symbols, _ = infer.get_classification_from_probabilities(predictions, "cnn")
        
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

