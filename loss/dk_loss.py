# This file contains the loss calculation function that is specified in the paper

from itertools import product
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

from graph.knowledge_graph import *

# labels for siftflow
siftflow_labels = ["awning", "balcony", "bird", "boat", "bridge", "building", "bus", \
          "car", "cow", "crosswalk", "desert", "door", "fence", "field", \
          "grass", "moon", "mountain", "person", "plant", "pole", "river", \
          "road", "rock", "sand", "sea", "sidewalk", "sign", "sky", \
          "staircase", "streetlight", "sun", "tree", "window"]
          
kgraph = CN_based_KnowledgeGraph(siftflow_labels, 0.15, 100, '/content/drive/My Drive/thesis/siftflow_similarity.txt')

def dk_distance(y_neighbor, y):
    return y_neighbor * K.log(tf.math.divide_no_nan(y_neighbor,y)) + (1-y_neighbor) * K.log(tf.math.divide_no_nan((1-y_neighbor),(1-y)))

def get_class_similarity(pred, kgraph):
    pred = tf.cast(pred, tf.int64)
    
    res = tf.gather_nd(kgraph.get_similarity(), pred)
    res = tf.cast(res, tf.float32)
    return res

def recursive_map(pred):
    if K.ndim(pred) > 2:
        return K.map_fn(lambda x: recursive_map(x), pred, dtype=tf.float32)
    else:
        return get_class_similarity(pred)

def dk_loss(y_true, y_pred):
    img_height = y_true.shape[1]
    img_width = y_true.shape[2]

    id_y_pred = K.argmax(y_pred, axis=3)
    pred_y_pred = K.max(y_pred, axis=3)
    true_y_true = tf.cast(K.max(y_true, axis=3), dtype=tf.int32)
    s = 0
    count = 0
    # Compute neighboring pixel loss contributions
    for i, j in product((-1, 0, 1), repeat=2):
        # if i == j == 0: continue
        # Take sliced image
        sliced_id_y_pred = id_y_pred[:, 1:-1, 1:-1]
        sliced_y_true = true_y_true[:, 1:-1, 1:-1]
        sliced_y_pred = pred_y_pred[:, 1:-1, 1:-1]

        # Take "shifted" image
        displaced_y_true = true_y_true[:, 1 + i:img_width - 1 + i, 1 + j:img_height - 1 + j]
        displaced_y_pred = pred_y_pred[:, 1 + i:img_width - 1 + i, 1 + j:img_height - 1 + j]
        displaced_id_y_pred = id_y_pred[:, 1+i:img_width - 1 + i, 1 + j:img_height - 1 + j]

        diff = sliced_y_true - displaced_y_true
        mask_t = tf.cast(K.equal(0, diff), dtype=tf.float32)
        mask_f = tf.cast(K.not_equal(0, diff), dtype=tf.float32)
        
        # choice 1
        c1 = tf.multiply(dk_distance(displaced_y_pred, sliced_y_pred), mask_t)

        # choice 2
        stacked_ids = tf.stack([displaced_id_y_pred, sliced_id_y_pred], axis=3)
        simi = K.map_fn(lambda x: recursive_map(x), stacked_ids, dtype=tf.float32)
        c2 = tf.multiply(K.relu(3-tf.multiply(dk_distance(displaced_y_pred, sliced_y_pred), simi)), mask_f)

        s += K.sum(c1 + c2)
    
        # s += c

        count += 1
    a = tf.cast(count * 128 * 128, dtype=tf.float32)
    s = s / a
    return s
