import numpy as np
from sklearn.metrics import confusion_matrix

# Add IoU score ca
# culation to the prediction comparation of the testing data
# return in percentage
def check_IoU(prediction, ground_true):
#     print(prediction)
#     print(ground_true)
    intersection = np.logical_and(prediction, ground_true)
    union = np.logical_or(prediction, ground_true)
    IoU_score = np.sum(intersection)/np.sum(union) * 100
#     print(intersection)
#     print(union)
    return IoU_score

def check_ca(prediction, ground_true):
    f_prediction = prediction.flatten()
    f_ground_true = ground_true.flatten()
    cm = confusion_matrix(f_prediction, f_ground_true)
    cm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    ca = np.sum(cm.diagonal()) / cm.diagonal().shape[0]
    return ca

def check_pa(prediction, ground_true):
    return (prediction == ground_true).mean()