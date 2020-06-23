import numpy as np

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