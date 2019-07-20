"""YOLO_v3 Model Defined in Keras."""

from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from yolov3.utils import compose
import numpy as np


def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1, 1)),
                DarknetConv2D_BN_Leaky(num_filters, (3, 3)))(x)
        x = Add()([x,y])
    return x


def darknet_body(x):
    """Darknent body having 52 Convolution2D layers"""
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    b1 = resblock_body(x, 256, 8)
    b2 = resblock_body(b1, 512, 8)
    b3 = resblock_body(b2, 1024, 4)
    return b1, b2, b3


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)),
            DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            DarknetConv2D_BN_Leaky(num_filters, (1, 1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3)),
            DarknetConv2D(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(input, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    b1, b2, b3 = darknet_body(input)

    x, y1 = make_last_layers(b1, 512, num_anchors * (num_classes + 5))
    x = compose(
            DarknetConv2D_BN_Leaky(256, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, b2.output])

    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x, b3])

    _, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(input, [y1, y2, y3])


def convert_box_params(b_xy, b_wh):
    b_min = b_xy - (b_wh / 2.0)
    b_max = b_xy + (b_wh / 2.0)
    b_min_max = K.concatenate([
        b_min[..., ::-1],  # y_min, x_min
        b_max[..., ::-1],  # y_max, x_max
    ])
    return b_min_max


# Lambda layer for postprocessing YOLOv3 output

def make_decoder_layer(all_anchors, num_classes, input_shape):

    def decode(yolo_outputs):
        num_scales = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_scales == 3 else [[3, 4, 5], [1, 2, 3]]

        b_min_max_list = []
        box_confidence_list = []
        class_probs_list = []

        for scale_idx in np.arange(3):
            anchors = all_anchors[anchor_mask[scale_idx]]
            output = yolo_outputs[scale_idx]
            num_anchors = len(anchors)

            batch_size = K.shape(output)[0]
            grid_shape = K.shape(output)[1:3]
            grid_height = grid_shape[0]  # height
            grid_width = grid_shape[1]  # width

            # reshape to tensor of dimensions batch_size, grid_height, grid_width, num_anchors, 5 + num_classes
            # the five box parameters are:
            #   t_x, t_y determine the center point of the box
            #   t_w, t_h determine the width and height of the box
            #   the box confidence indicates the confidence that box contains an object and box is accurate
            output = K.reshape(output, [-1, grid_height, grid_width, num_anchors, 5 + num_classes])

            # compute b_x, b_y for each cell and each anchor
            c_x = K.tile(K.reshape(K.arange(grid_width),  [1, -1, 1, 1]), [grid_height, 1,          num_anchors, 1])
            c_y = K.tile(K.reshape(K.arange(grid_height), [-1, 1, 1, 1]), [1,           grid_width, num_anchors, 1])
            c_xy = K.concatenate([c_x, c_y])
            c_xy = K.cast(c_xy, K.dtype(output))
            b_xy = (K.sigmoid(output[..., :2]) + c_xy) / K.cast(grid_shape[::-1], K.dtype(output))

            # compute b_w and b_h for each cell and each anchor
            p_wh = K.tile(K.reshape(K.constant(anchors), [1, 1, num_anchors, 2]), [grid_height, grid_width, 1, 1])
            b_wh = p_wh * K.exp(output[..., 2:4]) / K.cast(input_shape[::-1], K.dtype(output))

            b_min_max = K.reshape(convert_box_params(b_xy, b_wh), [batch_size, -1, 4])  # y_min, x_min, y_max, x_max

            # compute box confidence for each cell and each anchor
            box_confidence = K.reshape(K.sigmoid(output[..., 4]), [batch_size, -1])

            # compute class probabilities for each cell and each anchor
            class_probs = K.reshape(K.sigmoid(output[..., 5:]), [batch_size, -1, num_classes])

            b_min_max_list.append(b_min_max)
            box_confidence_list.append(box_confidence)
            class_probs_list.append(class_probs)

        return [
            K.concatenate(b_min_max_list, axis=1),
            K.concatenate(box_confidence_list, axis=1),
            K.concatenate(class_probs_list, axis=1)
        ]

    return Lambda(decode)


def tiny_yolo_body(input, num_anchors, num_classes):
    """Create Tiny YOLO_v3 model CNN body in keras."""
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3, 3)),
            MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3, 3)),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3, 3)),
            MaxPooling2D(pool_size=(2,2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3, 3)))(input)
    x2 = compose(
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1, 1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3, 3)),
            DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1, 1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3, 3)),
            DarknetConv2D(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])

    return Model(input, [y1, y2])
