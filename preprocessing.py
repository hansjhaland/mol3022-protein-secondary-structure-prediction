from one_hot_encodings import amino_acid_one_hot, secondary_structure_one_hot

def load_sequences_from_file(file_path):
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

def get_one_hot_encoding(sequence: str, one_hot_encoding: dict[str, list[int]]):
    """
    Return the one-hot encoding of the sequence.
    """
    one_hot_sequence = [one_hot_encoding[symbol] for symbol in sequence]
    return one_hot_sequence

def get_sequence_windows(sequence: list[list[int]], window_size: int, one_hot_encoding: dict[str, list[int]]):
    """
    Return the windows of the sequence.
    
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
        window: list[int] = sequence_copy[i:i+window_size]
        window = [symbol for encoding in window for symbol in encoding]
        windows.append(window)
    return windows

train_data_file = "data/protein-secondary-structure.train"
test_data_file = "data/protein-secondary-structure.test"

# NOTE: AA = Amino Acid
# NOTE: SS = Secondary Structure

# Read train and test data from files
train_AA_SS_seq_pairs = load_sequences_from_file(train_data_file)
test_AA_SS_seq_pairs = load_sequences_from_file(test_data_file)

# Split into input and target sequences
AA_seq_train = [sequence[0] for sequence in train_AA_SS_seq_pairs]
SS_seq_train = [sequence[1] for sequence in train_AA_SS_seq_pairs]

AA_seq_test = [sequence[0] for sequence in test_AA_SS_seq_pairs]
SS_seq_test = [sequence[1] for sequence in test_AA_SS_seq_pairs]

# Convert to one-hot encodings
AA_seq_train_one_hot = [get_one_hot_encoding(sequence, amino_acid_one_hot) for sequence in AA_seq_train]
SS_seq_train_one_hot = [get_one_hot_encoding(sequence, secondary_structure_one_hot) for sequence in SS_seq_train]

AA_seq_test_one_hot = [get_one_hot_encoding(sequence, amino_acid_one_hot) for sequence in AA_seq_test]
SS_seq_test_one_hot = [get_one_hot_encoding(sequence, secondary_structure_one_hot) for sequence in SS_seq_test]
    
# Convert to windows, which are the input to the neural network
window_size = 13 # NOTE: Window size used in the slides

train_seq_windows = [get_sequence_windows(sequence, window_size, amino_acid_one_hot) for sequence in AA_seq_train_one_hot]
test_seq_windows = [get_sequence_windows(sequence, window_size, amino_acid_one_hot) for sequence in AA_seq_test_one_hot]

X_train = [window for sequence in train_seq_windows for window in sequence]
y_train = [label for sequence in SS_seq_train_one_hot for label in sequence]

X_test = [window for sequence in test_seq_windows for window in sequence]
y_test = [label for sequence in SS_seq_test_one_hot for label in sequence]

print(len(X_train), len(y_train))

total_amino_acids = sum([len(sequence) for sequence in AA_seq_train_one_hot])
total_secondary_structures = sum([len(sequence) for sequence in SS_seq_train_one_hot])
print(total_amino_acids, total_secondary_structures)

 