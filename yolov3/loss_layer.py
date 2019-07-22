from keras import backend as K
from keras.layers import Lambda


def make_loss_layer():

    def compute_loss(tensors):
        y1, y2, y3, y1_true, y2_true, y3_true = tuple(tensors)

        return K.sum(y1_true) + K.sum(y2_true) + K.sum(y3_true)

    return Lambda(compute_loss)

