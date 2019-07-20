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
