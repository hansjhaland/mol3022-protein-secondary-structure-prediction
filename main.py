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