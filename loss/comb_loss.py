import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy

gamma = 1

def combination_loss(y_true, y_pred):
    class_num = y_true.shape[3]

    # D(x) = [1 − P(y = fake|x)]
    dx = tf.math.abs(y_pred[:, :, :, class_num-1] - y_true[:, :, :, class_num-1])

    # check if y_true is fake
    return tf.sum(tf.math.log(dx)) + \
           gamma * categorical_crossentropy(y_true[:, :, :, class_num], y_pred[:, :, :, class_num])

def generator_loss(y_true, y_pred):
    class_num = y_true.shape[3]
    dx = tf.math.abs(y_pred[:, :, :, class_num-1] - y_true[:, :, :, class_num-1])

    return tf.sum(tf.math.log(dx))