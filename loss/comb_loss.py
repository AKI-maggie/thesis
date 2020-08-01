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
    gan_mask = y_true[:, :, :, -1]

    # find mask
    fake_mask = tf.cast(K.equal(0, gan_mask), dtype=tf.float32)
    label_mask = tf.cast(K.equal(1, gan_mask), dtype=tf.float32)
    unlabel_mask = tf.cast(K.equal(2, gan_mask), dtype=tf.float32)

    # calculate for unlabel data


    # calculate for fake data


    # calculate for label data
    

    # mask = y_true[:, :, :, -1]
    # sum = 0
    # # fake data
    # sum -= tf.math.divide_no_nan(tf.reduce_sum(tf.math.log(y_pred[:, :, :, -1] * tf.cast((mask == 1), tf.float32))), tf.cast(tf.math.count_nonzero(mask==1), tf.float32))
    # # unlabeled data
    # sum -= tf.math.divide_no_nan(tf.reduce_sum(tf.math.log(1-y_pred[:, :, :, -1] * tf.cast((mask == -1), tf.float32))), tf.cast(tf.math.count_nonzero(mask==-1), tf.float32))
    # # K.print_tensor(sum, message='sum = ')
    # # labeled data
    # labeled_ypred = y_pred[:, :, :, :-1][mask == 0]
    # labeled_ytrue = y_true[:, :, :, :-1][mask == 0]
    sum = gamma * categorical_crossentropy(y_true, y_pred, from_logits=True)
    # K.print_tensor(sum, message='sum = ')
    sum = Add()([sum, 0.01 * dk_loss(y_true[:, :, :, :-1], y_pred[:, :, :, :-1])])
    return sum
