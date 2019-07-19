import cv2
import os
import colorsys
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from yolov3.model import make_decoder_layer
from yolov3.utils import make_boxed_image


class ObjectDetector(object):
    _defaults = {
        'model_path': 'model_data/yolov3.h5',
        'anchors_path': 'model_data/coco_anchors.txt',
        'classes_path': 'model_data/coco_classes.txt',
        'input_size': (416, 416),  # height, width
        'score': 0.7,  # a box is considered for class c iff confidence times class_prob for c is >= 0.7
        'iou': 0.4,  # boxes with iou 0.4 or greater are suppressed in nms
        'max_num_boxes' : 10  # max number of boxes for a class
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set values to up default values
        self.__dict__.update(kwargs)  # update values with user inputs
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
        self.anchors = self._get_anchors()
        self.num_anchors = self.anchors.shape[0]
        self.colors = self._get_colors()
        self.num_anchors_per_scale = 3
        self.num_scales = 3

        assert self.num_anchors == self.num_scales * self.num_anchors_per_scale, 'Mismatch of number of anchors'
        self.model = self._get_detection_model()

    def _get_class_names(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_colors(self):
        # generate colors for drawing bounding boxes
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        np.random.seed(1001)  # fix seed for consistent colors across runs
        np.random.shuffle(colors)  # shuffle colors
        np.random.seed(None)  # reset seed to default
        return colors

    def _get_detection_model(self):

        yolo_model = load_model(self.model_path, compile=False)

        assert len(yolo_model.output) == self.num_scales, \
            'Mismatch between model number of scales and given number of scales.'

        for idx in np.arange(self.num_scales):
            assert yolo_model.output[idx].shape[-1] == self.num_anchors_per_scale * (self.num_classes + 5), \
                'Mismatch between model output length and number of anchors and and number of classes'

        input = Input(shape=(self.input_size[0], self.input_size[1], 3))
        x1 = yolo_model(input)
        decoder_layer = make_decoder_layer(self.anchors, self.num_classes, self.input_size)
        x2 = decoder_layer(x1)

        return Model(input, x2)

    def detect_image(self, image):

        input_image, scale, offset_height, offset_width = make_boxed_image(image, self.input_size)

        input = np.array(input_image, dtype='float32')
        input /= 255.
        input = np.expand_dims(input, 0)  # add batch dimension.

        outputs = self.model.predict(input)

        # self.show_all_bounding_boxes(input_image, outputs)
        self.show_non_suppressed_bounding_boxes(input_image, outputs)

    def show_all_bounding_boxes(self, image, outputs, score_threshold=0.7):
        print('all bounding boxes')
        # the second value is 0 because batch size = 1 here for prediction
        boxes = outputs[0][0]
        confidence = np.reshape(outputs[1][0], [-1, 1])
        class_probs = outputs[2][0]
        scores = confidence * class_probs
        num_boxes = boxes.shape[0]

        annotated_image = image[...]

        font = cv2.FONT_HERSHEY_PLAIN

        for box_idx in np.arange(num_boxes):
            y_min = int(self.input_size[0] * boxes[box_idx, 0])
            x_min = int(self.input_size[1] * boxes[box_idx, 1])
            y_max = int(self.input_size[0] * boxes[box_idx, 2])
            x_max = int(self.input_size[1] * boxes[box_idx, 3])

            for class_index in np.arange(self.num_classes):  # self.num_classes):
                if scores[box_idx, class_index] >= score_threshold:
                    label = '{} {:.2f}'.format(self.class_names[class_index], class_probs[box_idx, class_index])
                    label_size = cv2.getTextSize(label, font, 1, 1)

                    label_width = label_size[0][0]
                    label_height = label_size[0][1]

                    cv2.rectangle(annotated_image,
                                  (x_min, y_min), (x_max, y_max),
                                  self.colors[class_index],
                                  2)
                    cv2.rectangle(annotated_image,
                                  (x_min, y_min - label_height),
                                  (x_min + label_width, y_min),
                                  self.colors[class_index],
                                  -1)
                    cv2.putText(annotated_image,
                                label,
                                (x_min, y_min),
                                font, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('all bounding boxes', annotated_image)

    def show_non_suppressed_bounding_boxes(self, image, outputs, max_num_boxes=10, score_threshold=0.7, iou_threshold=0.4):
        # the second value is 0 because batch size = 1 here for prediction
        boxes = outputs[0][0]
        confidence = outputs[1][0]
        class_probs = outputs[2][0]
        scores = np.reshape(confidence, [-1, 1]) * class_probs

        annotated_image = image[...]
        font = cv2.FONT_HERSHEY_PLAIN

        for class_index in np.arange(self.num_classes):

            pick_for_class = \
                non_max_suppression(boxes, scores[:, class_index], max_num_boxes, score_threshold, iou_threshold)

            for box_idx in pick_for_class:

                y_min = int(self.input_size[0] * boxes[box_idx, 0])
                x_min = int(self.input_size[1] * boxes[box_idx, 1])
                y_max = int(self.input_size[0] * boxes[box_idx, 2])
                x_max = int(self.input_size[1] * boxes[box_idx, 3])

                label = '{} {:.2f}'.format(self.class_names[class_index], class_probs[box_idx, class_index])
                label_size = cv2.getTextSize(label, font, 1, 1)

                label_width = label_size[0][0]
                label_height = label_size[0][1]

                cv2.rectangle(annotated_image,
                              (x_min, y_min), (x_max, y_max),
                              self.colors[class_index],
                              2)
                cv2.rectangle(annotated_image,
                              (x_min, y_min - label_height),
                              (x_min + label_width, y_min),
                              self.colors[class_index],
                              -1)
                cv2.putText(annotated_image,
                            label,
                            (x_min, y_min),
                            font, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('non suppressed bounding boxes', annotated_image)


def non_max_suppression(boxes, scores, max_num_boxes, score_threshold, iou_threshold):

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    y_min = boxes[:, 0]
    x_min = boxes[:, 1]
    y_max = boxes[:, 2]
    x_max = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x_max - x_min) * (y_max - y_min)

    # sort the indexes; note: one could use a priority queue of size max_num_boxes, but that's probably overkill
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        if scores[i] < score_threshold:
            return pick

        pick.append(i)

        if len(pick) == max_num_boxes:
            return pick

        # compute the width and height of the intersection of
        # the picked bounding box with all other bounding boxes
        yy_min = np.maximum(y_min[i], y_min[idxs[:last]])
        xx_min = np.maximum(x_min[i], x_min[idxs[:last]])
        yy_max = np.minimum(y_max[i], y_max[idxs[:last]])
        xx_max = np.minimum(x_max[i], x_max[idxs[:last]])
        w = np.maximum(0, xx_max - xx_min)
        h = np.maximum(0, yy_max - yy_min)

        # compute intersection over union
        iou = (w * h) / (area[i] + area[idxs[:last]] - w * h + 1e-5)

        # delete all indexes from the index list that have iou greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > iou_threshold)[0])))

    return pick


def main():
    filename = 'test_images/military_trucks.jpg'
    image = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    # cv2.imshow('original image', image)
    detector = ObjectDetector()
    detector.detect_image(image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
