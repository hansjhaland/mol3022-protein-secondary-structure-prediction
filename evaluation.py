import numpy as np
import matplotlib.pyplot as plt
import preprocessing as pp

def get_confusion_matrix(predicted_one_hots, target_one_hots) -> list[list[int]]:
    confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]
    for prediction, target in zip(predicted_one_hots, target_one_hots):
        target_index = np.argmax(target)
        prediction_index = np.argmax(prediction)
        confusion_matrix[target_index][prediction_index] += 1
    return confusion_matrix

def get_sensitivity_and_specificity_from_confusion_matrix(confusion_matrix) -> tuple[tuple[float],tuple[float],tuple[float]]:
    # NOTE: Sensitivity <=> Recall
    # NOTE: Specificity <=> Precision
    
    sensitivity_helix = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2])
    sensitivity_sheet = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1] + confusion_matrix[1][2])
    sensitivity_coil = confusion_matrix[2][2] / (confusion_matrix[2][0] + confusion_matrix[2][1] + confusion_matrix[2][2])
    
    specificity_helix = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0] + confusion_matrix[2][0])
    specificity_sheet = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1] + confusion_matrix[2][1])
    specificity_coil = confusion_matrix[2][2] / (confusion_matrix[0][2] + confusion_matrix[1][2] + confusion_matrix[2][2])
    
    return (sensitivity_helix, specificity_helix), (sensitivity_sheet, specificity_sheet), (sensitivity_coil, specificity_coil)
    

def plot_secondary_structure_distribution_in_training_set():
    training_set = "data/protein-secondary-structure.train"
    sequence_pairs = pp.load_sequences_from_file(training_set)
    secondary_structures = [pair[1] for pair in sequence_pairs]
    
    num_alpha_helix = 0
    num_beta_sheet = 0
    num_random_coil = 0
    
    for sequence in secondary_structures:
        for symbol in sequence:
            symbol = symbol.lower()
            if symbol == 'h':
                num_alpha_helix += 1
            elif symbol == 'e':
                num_beta_sheet += 1
            elif symbol == '_':
                num_random_coil += 1
                
    labels = ['Helix', 'Sheet', 'Coil']
    counts = [num_alpha_helix, num_beta_sheet, num_random_coil]
    
    plt.bar(labels, counts)
    plt.show()


def plot_sensitivity_and_specificity_per_class(model_measures: list[tuple[float]]):
    labels = ['Helix', 'Sheet', 'Coil']
    sensitivity = [x[0] for x in model_measures]
    specificity = [x[1] for x in model_measures]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, sensitivity, width, label='Sensitivity')
    rects2 = ax.bar(x + width/2, specificity, width, label='Specificity')
    
    ax.set_ylabel('Scores')
    # ax.set_title('Sensitivity and Specificity per class')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.ylim(0, 1.2)
    plt.show()
    
    
if __name__ == "__main__":
    plot_secondary_structure_distribution_in_training_set()

    