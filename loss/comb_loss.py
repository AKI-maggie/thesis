import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from loss.dk_loss import dk_loss
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Add


gamma = 1

# def combination_loss(y_true, y_pred):
#     class_num = y_true.shape[3]

#     # D(x) = [1 − P(y = fake|x)]
#     dx = tf.math.abs(y_pred[:, :, :, class_num-1] - y_true[:, :, :, class_num-1])

#     # check if y_true is fake
#     return tf.sum(tf.math.log(dx)) + \
#            gamma * categorical_crossentropy(y_true[:, :, :, class_num], y_pred[:, :, :, class_num])

def gan_activation(output):
    logexpsum = K.sum(K.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

def generator_loss(y_true, y_pred):
    class_num = y_true.shape[3]
    dx = tf.math.abs(y_pred[:, :, :, class_num-1] - y_true[:, :, :, class_num-1])

    return tf.sum(tf.math.log(dx))


def combination_loss(y_true, y_pred):
    sum = gamma * categorical_crossentropy(y_true, y_pred, from_logits=True)
    # K.print_tensor(sum, message='sum = ')
    sum = Add()([sum, 0.01 * dk_loss(y_true, y_pred)])
    return sum
