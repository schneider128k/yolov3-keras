import cv2
import os
import colorsys
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from yolov3.decoder_layer import make_decoder_layer
import time
import argparse


class ObjectDetector(object):
    _defaults = {
        'model_path': 'model_data/yolov3.h5',
        'anchors_path': 'model_data/coco_anchors.txt',
        'classes_path': 'model_data/coco_classes.txt',
        'height': 416,  # height
        'width' : 416,
        'score_threshold': 0.7,  # a box is considered for class c iff confidence times class_prob for c is >= 0.7
        'iou_threshold': 0.4,  # boxes with iou 0.4 or greater are suppressed in nms
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

        input = Input(shape=(self.height, self.width, 3))
        x1 = yolo_model(input)
        decoder_layer = make_decoder_layer(self.anchors, self.num_classes, (self.height, self.width))
        x2 = decoder_layer(x1)

        return Model(input, x2)

    def detect_image(self, image, nms=True):
        self.make_model_input(image)
        self.run_yolov3()
        if nms:
            self.show_nms_bounding_boxes()
        else:
            self.show_all_bounding_boxes()

    def make_model_input(self, image):
        self.image = image
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.scale = min(self.height / self.image_height, self.width / self.image_width)
        tmp_height = int(self.scale * self.image_height)
        tmp_width = int(self.scale * self.image_width)
        self.offset_height = (self.height - tmp_height) // 2
        self.offset_width = (self.width - tmp_width) // 2
        input = np.zeros((self.height, self.width, 3), np.uint8)
        input[:, :] = (128, 128, 128)
        input[self.offset_height:self.offset_height + tmp_height, self.offset_width:self.offset_width + tmp_width] = \
            cv2.resize(image, (tmp_width, tmp_height))

        input = np.array(input, dtype='float32')
        input /= 255.
        self.input = np.expand_dims(input, 0)  # add batch dimension.

    def run_yolov3(self):
        outputs = self.model.predict(self.input)
        # the second value is 0 because batch size = 1 here for prediction
        self.boxes = outputs[0][0]
        self.confidence = np.reshape(outputs[1][0], [-1, 1])
        self.class_probs = outputs[2][0]
        self.scores = self.confidence * self.class_probs

    def translate_coord(self, box):
        # the YOLOv3 model returns y and x coords in the range [0, 1] with respect to the the model height and width
        # they need to be translated to the coords of the original image
        y_min = int((box[0] * self.height - self.offset_height) / self.scale)
        x_min = int((box[1] * self.width - self.offset_width) / self.scale)
        y_max = int((box[2] * self.height- self.offset_height) / self.scale)
        x_max = int((box[3] * self.width - self.offset_width) / self.scale)
        return y_min, x_min, y_max, x_max

    def show_all_bounding_boxes(self):
        print('all bounding boxes')

        num_boxes = self.boxes.shape[0]
        print('num_boxes:', num_boxes)

        font = cv2.FONT_HERSHEY_PLAIN

        for box_idx in np.arange(num_boxes):

            y_min, x_min, y_max, x_max = self.translate_coord(self.boxes[box_idx])

            for class_index in np.arange(self.num_classes):

                if self.scores[box_idx, class_index] >= self.score_threshold:
                    label = '{} {:.2f}'.format(self.class_names[class_index], self.class_probs[box_idx, class_index])
                    label_size = cv2.getTextSize(label, font, 1, 1)

                    label_width = label_size[0][0]
                    label_height = label_size[0][1]

                    cv2.rectangle(self.image,
                                  (x_min, y_min), (x_max, y_max),
                                  self.colors[class_index],
                                  2)
                    cv2.rectangle(self.image,
                                  (x_min, y_min - label_height),
                                  (x_min + label_width, y_min),
                                  self.colors[class_index],
                                  -1)
                    cv2.putText(self.image,
                                label,
                                (x_min, y_min),
                                font, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('all bounding boxes', self.image)
        cv2.waitKey(1)

    def show_nms_bounding_boxes(self):

        font = cv2.FONT_HERSHEY_PLAIN

        for class_index in [0, 2, 3, 4, 7, 24]:  # np.arange(self.num_classes):

            pick_for_class = \
                non_max_suppression(self.boxes, self.scores[:, class_index], self.max_num_boxes, self.score_threshold, self.iou_threshold)

            for box_idx in pick_for_class:
                y_min, x_min, y_max, x_max = self.translate_coord(self.boxes[box_idx])

                label = '{} {:.2f}'.format(self.class_names[class_index], self.class_probs[box_idx, class_index])
                label_size = cv2.getTextSize(label, font, 1, 1)

                label_width = label_size[0][0]
                label_height = label_size[0][1]

                cv2.rectangle(self.image,
                              (x_min, y_min), (x_max, y_max),
                              self.colors[class_index],
                              2)
                cv2.rectangle(self.image,
                              (x_min, y_min - label_height),
                              (x_min + label_width, y_min),
                              self.colors[class_index],
                              -1)
                cv2.putText(self.image,
                            label,
                            (x_min, y_min),
                            font, 1, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('nms bounding boxes', self.image)
        cv2.waitKey(1)


# turn this into a class function ???
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


def main(path, video=False):

    detector = ObjectDetector()

    if video:
        video_file_path = path

        video_object = cv2.VideoCapture(video_file_path)
        frame_rate = video_object.get(5)  # frame rate
        print ("\nPlaying Video from", video_file_path, "with framerate", frame_rate)

        start = time.time()

        while True:
            frame_id = video_object.get(1)  # current frame number
            ret, frame = video_object.read()
            if ret is False or frame is None:
                break

            if frame_rate <= 31:
                if frame_id % 2 == 0:
                    detector.detect_image(frame)   # every second frame if frame rate low
            else:
                if frame_id % 4 == 0:
                    detector.detect_image(frame)   # every second frame if frame rate high

        video_object.release()
        end = time.time()
        print("Time elapsed in minutes: ", ((end - start) / 60))
    else:
        image = cv2.imread(path, cv2.COLOR_BGR2RGB)
        detector.detect_image(image)
        cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--video', default=None, help="video path"
    )

    parser.add_argument(
        '--image', default=None, help="image path"
    )

    args = parser.parse_args()

    if args.video:
        main(args.video, video=True)
    else:
        main(args.image)
