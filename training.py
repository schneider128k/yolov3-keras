import numpy as np
from keras.layers import Input
from keras import Model
from yolov3.model import make_yolo_model
from yolov3.loss_layer import make_loss_layer


class Trainer(object):

    def __init__(self):
        # hard code for now
        self.height = 416
        self.width = 416
        self.num_anchors_per_scale = 3
        self.num_classes = 4
        self.model = self.create_model()

    def create_model(self):
        input = Input(shape=(self.height, self.width, 3))

        yolo_model = make_yolo_model(input, self.num_anchors_per_scale, self.num_classes)

        y_true = [
           Input(shape=(self.height // factor, self.width // factor, 3)) for factor in [32, 16, 8]
        ]

        loss_layer = make_loss_layer()

        return Model(
            [input, *y_true],
            loss_layer([*yolo_model(input), *y_true])
        )


def main():
    trainer = Trainer()
    input = np.expand_dims(np.full((416, 416, 3), 1.0, dtype='float32'), 0)
    y_true_1 = np.expand_dims(np.full((13, 13, 3), 1.0, dtype='float32'), 0)
    y_true_2 = np.expand_dims(np.full((26, 26, 3), 1.0, dtype='float32'), 0)
    y_true_3 = np.expand_dims(np.full((52, 52, 3), 1.0, dtype='float32'), 0)

    loss = trainer.model.predict([input, y_true_1, y_true_2, y_true_3])

    print('loss:', loss[0])


if __name__ == "__main__":
    main()


