import numpy as np
from one_hot_encodings import amino_acid_one_hot, secondary_structure_one_hot

def load_sequences_from_file(file_path: str) -> list[tuple[str, str]]:
    """
    Read the file and return a list of amino acid and secondary structure sequence pairs.
    """
    # Open the file and read lines
    with open(file_path, "r") as f:
        lines = f.readlines()
        # Remove white space from lines
        lines = [line.strip() for line in lines]
        
    FILE_SEQUENCES_START_LINE = 8
    file_sequences = lines[FILE_SEQUENCES_START_LINE:]

    input_target_pairs = []
    amino_acid_sequence = []
    secondary_structure_sequence = []
    for line in file_sequences:
        if line == "<>":
            amino_acid_sequence = "".join(amino_acid_sequence)
            secondary_structure_sequence = "".join(secondary_structure_sequence)
            input_target_pairs.append((amino_acid_sequence, secondary_structure_sequence))
            amino_acid_sequence = []
            secondary_structure_sequence = []
            continue
        line_symbols = line.split(" ")
        if len(line_symbols) == 1:
            continue
        amino_acid_sequence.append(line_symbols[0])
        secondary_structure_sequence.append(line_symbols[1])
    input_target_pairs = input_target_pairs[1:]
    return input_target_pairs


def get_one_hot_encoding_of_sequence(sequence: str, one_hot_encoding: dict[str, list[int]]) -> list[list[int]]:
    """
    Return the one-hot encoding of a given sequence.
    """
    one_hot_sequence = [one_hot_encoding[symbol] for symbol in sequence]
    return one_hot_sequence


def get_concat_sequence_windows(sequence: list[list[int]], window_size: int, one_hot_encoding: dict[str, list[int]]) -> list[list[list[int|float]]]:
    """
    Return all windows  with the given window size for a given sequence.
    
    The sequence is assumed to be one-hot encoded.
    
    Each window is associated with one target value.
    
    Assumes that the symbol in the middle of the window is the "symbol of interest".
    To do this for the symbols at the start and end of the sequence, 
    the sequence is padded with paddings encoded as lists of zeros.
    """
    windows = []
    num_padding_symbols = window_size // 2
    sequence_copy = sequence.copy()
    for _ in range(num_padding_symbols):
            sequence_copy.insert(0, one_hot_encoding["pad"])
            sequence_copy.append(one_hot_encoding["pad"])
    for i in range(len(sequence_copy)-window_size+1):
        window = sequence_copy[i:i+window_size]
        window = [symbol for encoding in window for symbol in encoding]
        windows.append(window)
    return windows


def get_sequential_sequence_windows(sequence: list[list[int]], window_size: int, one_hot_encoding: dict[str, list[int]]):
    """
    Return all windows with the given window size for a given sequence, organized in a 2D array.
    """
    
    # NOTE: Only works properly when window_size is an ODD number.
    # For even window size, additional windows are added.
    windows = []
    num_padding_symbols = window_size // 2
    sequence_copy = sequence.copy()
    for _ in range(num_padding_symbols):
        sequence_copy.insert(0, one_hot_encoding["pad"])
        sequence_copy.append(one_hot_encoding["pad"])
    for i in range(len(sequence_copy)-window_size+1):
        window = sequence_copy[i:i+window_size]
        windows.append(window)
    return windows


def get_feedforward_amino_to_structure_data_sets(train_data_file: str, test_data_file: str, window_size=13) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Returns train and test datasets, containing pairs of amino acid sequence windows and target values.
    These data sets may be used as input and targets for a feedforward amino-to-structure model. 
    
    NOTE: AA = Amino Acid
    NOTE: SS = Secondary Structure
    
    NOTE: Window size 13 is used in the lecture slides
    """

    # Read train and test data from files
    train_AA_SS_seq_pairs = load_sequences_from_file(train_data_file)
    test_AA_SS_seq_pairs = load_sequences_from_file(test_data_file)

    # Split into input and target sequences
    AA_seq_train = [sequence[0] for sequence in train_AA_SS_seq_pairs]
    SS_seq_train = [sequence[1] for sequence in train_AA_SS_seq_pairs]

    AA_seq_test = [sequence[0] for sequence in test_AA_SS_seq_pairs]
    SS_seq_test = [sequence[1] for sequence in test_AA_SS_seq_pairs]
    
    sequence_lengths_train = [len(sequence) for sequence in AA_seq_train]
    sequence_lengths_test = [len(sequence) for sequence in AA_seq_test]

    # Convert to one-hot encodings
    AA_seq_train_one_hot = [get_one_hot_encoding_of_sequence(sequence, amino_acid_one_hot) for sequence in AA_seq_train]
    SS_seq_train_one_hot = [get_one_hot_encoding_of_sequence(sequence, secondary_structure_one_hot) for sequence in SS_seq_train]

    AA_seq_test_one_hot = [get_one_hot_encoding_of_sequence(sequence, amino_acid_one_hot) for sequence in AA_seq_test]
    SS_seq_test_one_hot = [get_one_hot_encoding_of_sequence(sequence, secondary_structure_one_hot) for sequence in SS_seq_test]
        
    # Convert to windows, which are the input to the neural network
    train_seq_windows = [get_concat_sequence_windows(sequence, window_size, amino_acid_one_hot) for sequence in AA_seq_train_one_hot]
    test_seq_windows = [get_concat_sequence_windows(sequence, window_size, amino_acid_one_hot) for sequence in AA_seq_test_one_hot]

    # Number of windows is equal to number of labels
    X_train = [window for sequence in train_seq_windows for window in sequence]
    y_train = [label for sequence in SS_seq_train_one_hot for label in sequence]

    X_test = [window for sequence in test_seq_windows for window in sequence]
    y_test = [label for sequence in SS_seq_test_one_hot for label in sequence]

    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test), np.asarray(sequence_lengths_train), np.asarray(sequence_lengths_test)


def get_feedforward_structure_to_structure_data_sets(train_data_file: str, test_data_file: str, window_size = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns train and test datasets, containing pairs of secondary structure sequence windows and target values.
    These data sets may be used as input and targets for a feedforward structure-to-structure model.
    
    NOTE: These inputs should only be used when training the model. 
    When doing inference, the inputs needs to be converted from the amino-to-structure outputs. 
    
    NOTE: SS = Secondary Structure

    """
    train_sequence_pairs = load_sequences_from_file(train_data_file)
    test_sequence_pairs = load_sequences_from_file(test_data_file)
    
    SS_seq_train = [sequence[1] for sequence in train_sequence_pairs]
    SS_seq_test = [sequence[1] for sequence in test_sequence_pairs]
    
    SS_seq_train_one_hot = [get_one_hot_encoding_of_sequence(sequence, secondary_structure_one_hot) for sequence in SS_seq_train]
    SS_seq_test_one_hot = [get_one_hot_encoding_of_sequence(sequence, secondary_structure_one_hot) for sequence in SS_seq_test]
    
    train_seq_windows = [get_concat_sequence_windows(sequence, window_size, secondary_structure_one_hot) for sequence in SS_seq_train_one_hot]
    test_seq_windows = [get_concat_sequence_windows(sequence, window_size, secondary_structure_one_hot) for sequence in SS_seq_test_one_hot]
    
    X_train = [window for sequence in train_seq_windows for window in sequence]
    y_train = [label for sequence in SS_seq_train_one_hot for label in sequence]
    
    X_test = [window for sequence in test_seq_windows for window in sequence]
    y_test = [label for sequence in SS_seq_test_one_hot for label in sequence]
    
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)


def get_sequential_data_set(train_data_file: str, test_data_file: str, window_size = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns train and test datasets, containing containing two-dimensional windows
    """
    train_AA_SS_seq_pairs = load_sequences_from_file(train_data_file)
    test_AA_SS_seq_pairs = load_sequences_from_file(test_data_file)

    # Split into input and target sequences
    AA_seq_train = [sequence[0] for sequence in train_AA_SS_seq_pairs]
    SS_seq_train = [sequence[1] for sequence in train_AA_SS_seq_pairs]

    AA_seq_test = [sequence[0] for sequence in test_AA_SS_seq_pairs]
    SS_seq_test = [sequence[1] for sequence in test_AA_SS_seq_pairs]
    
    # Convert to one-hot encodings
    AA_seq_train_one_hot = [get_one_hot_encoding_of_sequence(sequence, amino_acid_one_hot) for sequence in AA_seq_train]
    SS_seq_train_one_hot = [get_one_hot_encoding_of_sequence(sequence, secondary_structure_one_hot) for sequence in SS_seq_train]

    AA_seq_test_one_hot = [get_one_hot_encoding_of_sequence(sequence, amino_acid_one_hot) for sequence in AA_seq_test]
    SS_seq_test_one_hot = [get_one_hot_encoding_of_sequence(sequence, secondary_structure_one_hot) for sequence in SS_seq_test]
    
    train_seq_windows = [get_sequential_sequence_windows(sequence, window_size, amino_acid_one_hot) for sequence in AA_seq_train_one_hot]
    test_seq_windows = [get_sequential_sequence_windows(sequence, window_size, amino_acid_one_hot) for sequence in AA_seq_test_one_hot]
    
    
    X_train = [window for sequence in train_seq_windows for window in sequence]
    y_train = [label for sequence in SS_seq_train_one_hot for label in sequence]

    X_test = [window for sequence in test_seq_windows for window in sequence]
    y_test = [label for sequence in SS_seq_test_one_hot for label in sequence]
    
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)
    
    
def get_CNN_2D_data_set(train_data_file: str, test_data_file: str, window_size = 13) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform the two-dimensional windows in the sequential data set to their transpose.
    This seemed like a fitting representation for the 2D CNN model.
    """
    X_train, y_train, X_test, y_test = get_sequential_data_set(train_data_file, test_data_file, window_size)
    
    X_train = [np.transpose(window) for window in X_train]
    X_test = [np.transpose(window) for window in X_test]
    
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)
        

if __name__ == "__main__":
    
    train_data_file = "data/protein-secondary-structure.train"
    test_data_file = "data/protein-secondary-structure.test"  
    
    X_train, y_train, X_test, y_test, sequence_lengths_train, sequence_lengths_test = get_feedforward_amino_to_structure_data_sets(train_data_file, test_data_file, window_size=13)
    SS_X_train, SS_y_train, SS_X_test, SS_y_test = get_feedforward_structure_to_structure_data_sets(train_data_file, test_data_file, window_size=13)
    CNN_2D_X_train, CNN_2D_y_train, CNN_2D_X_test, CNN_2D_y_test = get_CNN_2D_data_set(train_data_file, test_data_file, window_size=13)
     
    print("Amino to Structure shapes:")
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    print(sequence_lengths_train.shape, sequence_lengths_test.shape)
    print()
    print("Structure to Structure shapes:")
    print(SS_X_train.shape, SS_y_train.shape)
    print(SS_X_test.shape, SS_y_test.shape)
    print()
    print("CNN 2D shapes")
    print(CNN_2D_X_train.shape, CNN_2D_y_train.shape)
    print(CNN_2D_X_test.shape, CNN_2D_y_test.shape)


    print_sequences = False
    if print_sequences:
        test_file_sequence_pairs = load_sequences_from_file(test_data_file)

        [print("Input:", pair[0], "\n", "Target:", pair[1], "\n") for pair in test_file_sequence_pairs[:10]]
        
    count_sequences = True
    if count_sequences:
        train_file_pairs = load_sequences_from_file(train_data_file)
        test_file_pairs = load_sequences_from_file(test_data_file)
        
        print(f"{len(train_file_pairs)} sequences in training set. {len(test_file_pairs)} sequences in test set.")
