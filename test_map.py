import tensorflow as tf
import numpy as np
import argparse
import colorsys
import cv2
import os

import yolo.config as cfg
from yolo.yolo_v2 import yolo_v2
# from yolo.darknet19 import Darknet19
from AIZOO_dataset import AIZOO_dataset
from utils.BoundingBox_class import BoundingBox
from utils.BoundingBoxes_class import BoundingBoxes
from utils.Evaluator import *

class Detector(object):
    def __init__(self, yolo, data, weights_file):
        self.yolo = yolo
        self.data = data
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.box_per_cell = cfg.BOX_PRE_CELL
        self.threshold = cfg.THRESHOLD
        self.anchor = cfg.ANCHOR

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restore weights from: ' + weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, weights_file)

    def detect(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        image = np.reshape(image, [1, self.image_size, self.image_size, 3])

        output = self.sess.run(self.yolo.logits, feed_dict = {self.yolo.images: image})

        results = self.calc_output(output)

        for i in range(len(results)):
            results[i][1] *= (1.0 * image_w / self.image_size)
            results[i][2] *= (1.0 * image_h / self.image_size)
            results[i][3] *= (1.0 * image_w / self.image_size)
            results[i][4] *= (1.0 * image_h / self.image_size)

        return results


    def calc_output(self, output):
        output = np.reshape(output, [self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        boxes = np.reshape(output[:, :, :, :4], [self.cell_size, self.cell_size, self.box_per_cell, 4])    #boxes coordinate
        boxes = self.get_boxes(boxes) * self.image_size

        confidence = np.reshape(output[:, :, :, 4], [self.cell_size, self.cell_size, self.box_per_cell])    #the confidence of the each anchor boxes
        confidence = 1.0 / (1.0 + np.exp(-1.0 * confidence))
        confidence = np.tile(np.expand_dims(confidence, 3), (1, 1, 1, self.num_classes))

        classes = np.reshape(output[:, :, :, 5:], [self.cell_size, self.cell_size, self.box_per_cell, self.num_classes])    #classes
        classes = np.exp(classes) / np.tile(np.expand_dims(np.sum(np.exp(classes), axis=3), axis=3), (1, 1, 1, self.num_classes))

        probs = classes * confidence

        filter_probs = np.array(probs >= self.threshold, dtype = 'bool')
        filter_index = np.nonzero(filter_probs)
        box_filter = boxes[filter_index[0], filter_index[1], filter_index[2]]
        probs_filter = probs[filter_probs]
        classes_num = np.argmax(filter_probs, axis = 3)[filter_index[0], filter_index[1], filter_index[2]]

        sort_num = np.array(np.argsort(probs_filter))[::-1]
        box_filter = box_filter[sort_num]
        probs_filter = probs_filter[sort_num]
        classes_num = classes_num[sort_num]

        for i in range(len(probs_filter)):
            if probs_filter[i] == 0:
                continue
            for j in range(i+1, len(probs_filter)):
                if self.calc_iou(box_filter[i], box_filter[j]) > 0.5:
                    probs_filter[j] = 0.0

        filter_probs = np.array(probs_filter > 0, dtype = 'bool')
        probs_filter = probs_filter[filter_probs]
        box_filter = box_filter[filter_probs]
        classes_num = classes_num[filter_probs]

        results = []
        for i in range(len(probs_filter)):
            results.append([self.classes[classes_num[i]], box_filter[i][0], box_filter[i][1],
                            box_filter[i][2], box_filter[i][3], probs_filter[i]])

        return results

    def get_boxes(self, boxes):
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                                         [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        boxes1 = np.stack([(1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 0])) + offset) / self.cell_size,
                           (1.0 / (1.0 + np.exp(-1.0 * boxes[:, :, :, 1])) + np.transpose(offset, (1, 0, 2))) / self.cell_size,
                           np.exp(boxes[:, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 5]) / self.cell_size,
                           np.exp(boxes[:, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 5]) / self.cell_size])

        return np.transpose(boxes1, (1, 2, 3, 0))


    def calc_iou(self, box1, box2):
        width = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        height = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])

        if width <= 0 or height <= 0:
            intersection = 0
        else:
            intersection = width * height

        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors


    def draw(self, image, result):
        image_h, image_w, _ = image.shape
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
            ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
            xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
            ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(image, result[i][0] + ':%.2f' % result[i][5], (xmin + 1, ymin + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            print(result[i][0], ':%.2f%%' % (result[i][5] * 100 ))


    def image_detect(self, imagename):
        image = cv2.imread(imagename)
        result = self.detect(image)
        print(result)
        self.draw(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(0)


    def Map_compute(self):
        self.results_all = []
        data_pairs = self.data.load_map_data('test')  # len:1839

        for i in range(len(data_pairs)):
            data_i = data_pairs[i]
            print(data_i)
            img_path = data_i['image_path']  #'face/face_mask',x1,y1,x2,y2
            image = cv2.imread(img_path)
            result = self.detect(image)

            result_true = []
            image_h, image_w, _ = image.shape
            for i in range(len(result)):
                xmin = max(int(result[i][1] - 0.5 * result[i][3]), 0)
                ymin = max(int(result[i][2] - 0.5 * result[i][4]), 0)
                xmax = min(int(result[i][1] + 0.5 * result[i][3]), image_w)
                ymax = min(int(result[i][2] + 0.5 * result[i][4]), image_h)
                result_true.append([result[i][0], xmin, ymin, xmax, ymax, result[i][5]])

            self.results_all.append({'image_path': img_path, 'label': result_true})
            print({'image_path': img_path, 'label': result_true})
            print('\n')
        print(len(self.results_all))

        allBoundingBoxes, allClasses = self.getBoundingBoxes(
            data_pairs, True, BBFormat.XYX2Y2, CoordinatesType.Absolute,
            imgSize=(self.image_size, self.image_size))

        allBoundingBoxes, allClasses = self.getBoundingBoxes(
            self.results_all, False, BBFormat.XYX2Y2, CoordinatesType.Absolute,
            allBoundingBoxes, allClasses, imgSize=(self.image_size, self.image_size))
        allClasses.sort()

        evaluator = Evaluator()
        acc_AP = 0
        validClasses = 0

        # Plot Precision x Recall curve
        detections = evaluator.PlotPrecisionRecallCurve(
            allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
            IOUThreshold=0.5,  # IOU threshold
            method=MethodAveragePrecision.ElevenPointInterpolation,
            showAP=True,  # Show Average Precision in the title of the plot
            showInterpolatedPrecision=True,  # Don't plot the interpolated precision curve
            savePath='',
            showGraphic=True)
        print(detections)
        # each detection is a class
        for metricsPerClass in detections:

            # Get metric values per each class
            cl = metricsPerClass['class']
            ap = metricsPerClass['AP']
            precision = metricsPerClass['precision']
            recall = metricsPerClass['recall']
            totalPositives = metricsPerClass['total positives']
            total_TP = metricsPerClass['total TP']
            total_FP = metricsPerClass['total FP']

            if totalPositives > 0:
                validClasses = validClasses + 1
                acc_AP = acc_AP + ap
                prec = ['%.2f' % p for p in precision]
                rec = ['%.2f' % r for r in recall]
                ap_str = "{0:.2f}%".format(ap * 100)
                # ap_str = "{0:.4f}%".format(ap * 100)
                print('AP: %s (%s)' % (ap_str, cl))

        mAP = acc_AP / validClasses
        mAP_str = "{0:.2f}%".format(mAP * 100)
        print('mAP: %s' % mAP_str)


    def getBoundingBoxes(self, data_set, isGT, bbFormat, coordType, allBoundingBoxes=None, allClasses=None, imgSize=(0, 0)):
        if allBoundingBoxes is None:
            allBoundingBoxes = BoundingBoxes()
        if allClasses is None:
            allClasses = []
        for i in range(len(data_set)):
            data_i = data_set[i]
            nameOfImage = data_i['image_path']
            nameOfImage = (nameOfImage.split('/')[-1]).split('.jpg')[0]
            for j in range(len(data_i['label'])):
                label_j = data_i['label'][j]
                if isGT:
                    idClass = label_j[0]
                    x1 = float(label_j[1])
                    y1 = float(label_j[2])
                    x2 = float(label_j[3])
                    y2 = float(label_j[4])
                    bb = BoundingBox(nameOfImage, idClass, x1, y1, x2, y2, coordType, imgSize, BBType.GroundTruth,
                                     format=bbFormat)

                else:
                    idClass = label_j[0]
                    x1 = float(label_j[1])
                    y1 = float(label_j[2])
                    x2 = float(label_j[3])
                    y2 = float(label_j[4])
                    confidence = float(label_j[5])
                    bb = BoundingBox(nameOfImage, idClass, x1, y1, x2, y2, coordType, imgSize, BBType.Detected,
                                     confidence, format=bbFormat)
                allBoundingBoxes.addBoundingBox(bb)
                if idClass not in allClasses:
                    allClasses.append(idClass)
        return allBoundingBoxes, allClasses



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'yolo_v2.ckpt-33000', type = str)    # darknet-19.ckpt
    parser.add_argument('--weight_dir', default = 'output', type = str)
    parser.add_argument('--data_dir', default = 'data', type = str)
    parser.add_argument('--gpu', default = '', type = str)    # which gpu to be selected
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'#args.gpu    # configure gpu
    weights_file = os.path.join(args.data_dir, args.weight_dir, args.weights)

    yolo = yolo_v2(False)    # 'False' mean 'test'
    # yolo = Darknet19(False)
    AIZOO = AIZOO_dataset()
    detector = Detector(yolo, AIZOO, weights_file)

    #detect the video
    #cap = cv2.VideoCapture('asd.mp4')
    #cap = cv2.VideoCapture(0)
    #detector.video_detect(cap)

    #detect the image
    imagename = './data/AIZOO_data/val/test_00004916.jpg'
    # detector.image_detect(imagename)
    detector.Map_compute()

if __name__ == '__main__':
    main()
