import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from loss.dk_loss import dk_loss

gamma = 1

# def combination_loss(y_true, y_pred):
#     class_num = y_true.shape[3]

#     # D(x) = [1 − P(y = fake|x)]
#     dx = tf.math.abs(y_pred[:, :, :, class_num-1] - y_true[:, :, :, class_num-1])

#     # check if y_true is fake
#     return tf.sum(tf.math.log(dx)) + \
#            gamma * categorical_crossentropy(y_true[:, :, :, class_num], y_pred[:, :, :, class_num])

def generator_loss(y_true, y_pred):
    class_num = y_true.shape[3]
    dx = tf.math.abs(y_pred[:, :, :, class_num-1] - y_true[:, :, :, class_num-1])

    return tf.sum(tf.math.log(dx))


def combination_loss(y_true, y_pred):
    mask = y_true[:, :, :, -1]
    sum = 0
    # fake data
    sum -= tf.math.divide_no_nan(tf.reduce_sum(tf.math.log(y_pred[:, :, :, -1] * tf.cast((mask == 1), tf.float32))), tf.cast(tf.math.count_nonzero(mask==1), tf.float32))
    # unlabeled data
    sum -= tf.math.divide_no_nan(tf.reduce_sum(tf.math.log(1-y_pred[:, :, :, -1] * tf.cast((mask == -1), tf.float32))), tf.cast(tf.math.count_nonzero(mask==-1), tf.float32))
    # labeled data
    labeled_ypred = y_pred[:, :, :, :-1][mask == 0]
    labeled_ytrue = y_true[:, :, :, :-1][mask == 0]
    sum += gamma * categorical_crossentropy(labeled_ytrue, labeled_ypred)

    return sum + 0.001 * dk_loss(y_true, y_pred)

