from one_hot_encodings import amino_acid_one_hot, secondary_structure_one_hot

def get_sequences_from_file(file_path):
    """
    Read the file and return the amino acid sequence and secondary structure sequence.
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

def get_one_hot_encoding(sequence, one_hot_encoding):
    """
    Return the one-hot encoding of the sequence.
    """
    one_hot_sequence = [one_hot_encoding[symbol] for symbol in sequence]
    return one_hot_sequence

def get_windows(sequence, window_size, one_hot_encoding):
    """
    Return the windows of the sequence.
    
    Assumes that the symbol in the middle of the window is the "symbol of interest".
    To do this for the symbols at the start and end of the sequence, 
    the sequence is padded with paddings encoded as lists of zeros.
    """
    windows = []
    num_padding_symbols = window_size // 2
    for i in range(len(sequence)):
        window: list[int] = sequence[i:i+window_size]
        for _ in range(num_padding_symbols):
            window.insert(0, one_hot_encoding["pad"])
            window.append(one_hot_encoding["pad"])
        windows.append(window)
    return windows

train_data_file = "data/protein-secondary-structure.train"
test_data_file = "data/protein-secondary-structure.test"

# Read train and test data from files
train_input_target_pairs = get_sequences_from_file(train_data_file)
test_input_target_pairs = get_sequences_from_file(test_data_file)

# Split into input and target sequences
X_train = [sequence[0] for sequence in train_input_target_pairs]
y_train = [sequence[1] for sequence in train_input_target_pairs]

X_test = [sequence[0] for sequence in test_input_target_pairs]
y_test = [sequence[1] for sequence in test_input_target_pairs]

# Convert to one-hot encodings
X_train_one_hot = [get_one_hot_encoding(sequence, amino_acid_one_hot) for sequence in X_train]
y_train_one_hot = [get_one_hot_encoding(sequence, secondary_structure_one_hot) for sequence in y_train]

X_test_one_hot = [get_one_hot_encoding(sequence, amino_acid_one_hot) for sequence in X_test]
y_test_one_hot = [get_one_hot_encoding(sequence, secondary_structure_one_hot) for sequence in y_test]
    
# Get windows
window_size = 13

X_train_windows = get_windows(X_train_one_hot, window_size, amino_acid_one_hot)

X_test_windows = get_windows(X_test_one_hot, window_size, amino_acid_one_hot)

print("done")
 