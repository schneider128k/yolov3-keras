from functools import reduce
import numpy as np
import cv2


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def make_boxed_image(image, new_size):
    height, width, _ = image.shape
    new_height, new_width = new_size
    scale = min(new_height / height, new_width / width)
    tmp_height = int(scale * height)
    tmp_width = int(scale * width)
    offset_height = (new_height - tmp_height) // 2
    offset_width = (new_width - tmp_width) // 2
    new_image = np.zeros((new_height, new_width, 3), np.uint8)
    new_image[:, :] = (128, 128, 128)
    new_image[offset_height:offset_height + tmp_height, offset_width:offset_width + tmp_width] = \
        cv2.resize(image, (tmp_width, tmp_height))
    # cv2.imshow('model image', new_image)
    return new_image, scale, offset_height, offset_width


