import numpy as np
from sklearn.metrics import confusion_matrix, jaccard_score

# Add IoU score ca
# culation to the prediction comparation of the testing data
# return in percentage
def check_IoU(prediction, ground_true):
    f_prediction = prediction.flatten()
    f_ground_true = ground_true.flatten()
#     print(prediction)
#     print(ground_true)
    return jaccard_score(f_ground_true, f_prediction, average='micro') * 100

def check_ca(prediction, ground_true):
    f_prediction = prediction.flatten()
    f_ground_true = ground_true.flatten()
    cm = confusion_matrix(f_prediction, f_ground_true)
    cm = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    ca = np.sum(cm.diagonal()) / cm.diagonal().shape[0]
    return ca * 100

def check_pa(prediction, ground_true):
    return (prediction == ground_true).mean() * 100