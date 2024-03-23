import numpy as np

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
    

# NOTE: Instances in training set
def get_number_of_secondary_structure_instances():
    pass

def plot_secondary_structure_distribution_in_training_set():
    pass