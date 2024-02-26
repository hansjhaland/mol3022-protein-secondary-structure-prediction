import tensorflow as tf
print("Tensorflow version:", tf.__version__)

from preprocessing import *

def get_seq_to_struct_model(input_length, num_classes) -> tf.keras.models.Sequential:
    input_length = input_length
    num_classes = num_classes
    
    tf.random.set_seed(42)
    
    seq_to_struct_model = tf.keras.models.Sequential()
    seq_to_struct_model.add(tf.keras.layers.Input(shape=(input_length,)))
    seq_to_struct_model.add(tf.keras.layers.Dropout(rate=0.2))
    seq_to_struct_model.add(tf.keras.layers.Dense(128, activation='relu'))
    seq_to_struct_model.add(tf.keras.layers.Dropout(rate=0.2))
    seq_to_struct_model.add(tf.keras.layers.Dense(64, activation='relu'))
    seq_to_struct_model.add(tf.keras.layers.Dropout(rate=0.2))
    seq_to_struct_model.add(tf.keras.layers.Dense(64, activation='relu'))
    seq_to_struct_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    seq_to_struct_model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['precision', 'recall']) # TODO: May just want to implement my own precision and recall after final predictions
    
    return seq_to_struct_model

def get_struct_to_struct_model(input_length, num_classes) -> tf.keras.models.Sequential:
    input_length = input_length
    num_classes = num_classes
    
    tf.random.set_seed(42)
    
    struct_to_struct_model = tf.keras.models.Sequential()
    struct_to_struct_model.add(tf.keras.layers.Input(shape=(input_length,)))
    struct_to_struct_model.add(tf.keras.layers.Dropout(rate=0.2))
    struct_to_struct_model.add(tf.keras.layers.Dense(128, activation='relu'))
    struct_to_struct_model.add(tf.keras.layers.Dropout(rate=0.2))
    struct_to_struct_model.add(tf.keras.layers.Dense(64, activation='relu'))
    struct_to_struct_model.add(tf.keras.layers.Dropout(rate=0.2))
    struct_to_struct_model.add(tf.keras.layers.Dense(64, activation='relu'))
    struct_to_struct_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    struct_to_struct_model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['precision', 'recall']) # TODO: May just want to implement my own precision and recall after final predictions
    
    return struct_to_struct_model


def get_struct_struct_model_input(seq_to_struct_model, amino_acid_one_hot, secondary_structure_one_hot, train_data_file) -> np.ndarray:
    sequence_pairs = load_sequences_from_file(train_data_file)
    AA_seq = [sequence[0] for sequence in sequence_pairs]
    AA_seq_one_hots = [get_one_hot_encoding(sequence, amino_acid_one_hot) for sequence in AA_seq]
    
    struct_windows = []
    for aa_seq_encoding in AA_seq_one_hots:
        aa_window = get_sequence_windows(aa_seq_encoding, 13, amino_acid_one_hot)
        aa_window = np.asarray(aa_window)
        struct_probabilities = seq_to_struct_model.predict(aa_window)
        ss_prob_window = get_sequence_windows(struct_probabilities.tolist(), 13, secondary_structure_one_hot)
        struct_windows.append(ss_prob_window)
        
    X_train_struct = [window for sequence in struct_windows for window in sequence]
    
    return np.asarray(X_train_struct)
    
    
    
if __name__ == "__main__":
    train_data_file = "data/protein-secondary-structure.train"
    test_data_file = "data/protein-secondary-structure.test" 
    
    save_seq_model = False
    save_model_path_seq = "pretrained/seq_to_struct_model.keras"
    save_struct_model = False
    save_model_path_struct = "pretrained/struct_to_struct_model.keras"
    
    load_seq_model = True
    load_struct_model = True
    
    X_train, y_train, X_test, y_test = get_data_sets_for_supervised_learning(train_data_file, test_data_file)
    
    if load_seq_model:
        seq_to_struct_model = tf.keras.models.load_model(save_model_path_seq)
        print(f"Model loaded from {save_model_path_seq}")
    else:
        seq_to_struct_model = get_seq_to_struct_model(X_train.shape[1], y_train.shape[1])
        seq_to_struct_model.fit(X_train, y_train, epochs=30, validation_split=0.1)
    
    if save_seq_model:
        seq_to_struct_model.save(save_model_path_seq)
        print(f"Model saved to {save_model_path_seq}")
        
    X_train_struct = get_struct_struct_model_input(seq_to_struct_model, amino_acid_one_hot, secondary_structure_one_hot, train_data_file)
    X_test_struct = get_struct_struct_model_input(seq_to_struct_model, amino_acid_one_hot, secondary_structure_one_hot, test_data_file)

    if load_struct_model:
        struct_to_struct_model = tf.keras.models.load_model(save_model_path_struct)
        print(f"Model loaded from {save_model_path_struct}")
    else:
        struct_to_struct_model = get_struct_to_struct_model(X_train_struct.shape[1], y_train.shape[1])
        struct_to_struct_model.fit(X_train_struct, y_train, epochs=30, validation_split=0.1)

    if save_struct_model:
        struct_to_struct_model.save(save_model_path_struct)
        print(f"Model saved to {save_model_path_struct}")
        
    y_pred = struct_to_struct_model.predict(X_test_struct)
    
    print(y_pred[0:10])