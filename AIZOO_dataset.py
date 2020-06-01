import os
import yolo.config as cfg
import numpy as np
from glob import glob
import pickle
import cv2
import xml.etree.ElementTree as ET
import copy


class AIZOO_dataset(object):
    def __init__(self, reload=False):
        self.data_dir = os.path.join(cfg.DATA_DIR,'AIZOO_data')
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.classes = cfg.CLASSES
        self.num_classes = len(self.classes)
        self.box_per_cell = cfg.BOX_PRE_CELL
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        self.reload = reload

        self.count=0
        self.epoch=1
        self.count_t=0


    def load_labels(self, model):
        if model == 'train':
            self.data_path = os.path.join(self.data_dir, 'train')
            image_list = glob('{}/*.jpg'.format(self.data_path))
        if model == 'test':
            self.data_path = os.path.join(self.data_dir, 'val')
            image_list = glob('{}/*.jpg'.format(self.data_path))
            print(len(image_list))

        labels = []
        for img_path in image_list:
            label, num = self.load_data(img_path)
            if num == 0:
                continue
            labels.append({'imagename': img_path, 'labels': label})
        np.random.shuffle(labels)
        return labels

    def load_data(self, img_path):
        print(img_path)
        img = cv2.imread(img_path)
        image_height = img.shape[0]
        image_width = img.shape[1]

        label = np.zeros([self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        path_no_suffix = img_path.split('.')[0]
        lab_path = path_no_suffix + '.xml'

        tree = ET.parse(lab_path)
        # image_size = tree.find('size')
        # image_width = float(image_size.find('width').text)
        # image_height = float(image_size.find('height').text)
        h_ratio = 1.0 * self.image_size / image_height
        w_ratio = 1.0 * self.image_size / image_width

        objects = tree.findall('object')
        for obj in objects:
            box = obj.find('bndbox')
            x1 = max(min((float(box.find('xmin').text)) * w_ratio, self.image_size), 0)
            y1 = max(min((float(box.find('ymin').text)) * h_ratio, self.image_size), 0)
            x2 = max(min((float(box.find('xmax').text)) * w_ratio, self.image_size), 0)
            y2 = max(min((float(box.find('ymax').text)) * h_ratio, self.image_size), 0)
            class_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [0.5 * (x1 + x2) / self.image_size, 0.5 * (y1 + y2) / self.image_size,
                     np.sqrt((x2 - x1) / self.image_size), np.sqrt((y2 - y1) / self.image_size)]
            cx = 1.0 * boxes[0] * self.cell_size
            cy = 1.0 * boxes[1] * self.cell_size
            xind = int(np.floor(cx))
            yind = int(np.floor(cy))

            label[yind, xind, :, 0] = 1
            label[yind, xind, :, 1:5] = boxes
            label[yind, xind, :, 5 + class_ind] = 1

        return label, len(objects)

    def next_batches(self, label):
        images = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
        labels = np.zeros([self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        num = 0
        while num < self.batch_size:
            imagename = label[self.count]['imagename']
            images[num, :, :, :] = self.image_read(imagename)
            labels[num, :, :, :, :] = label[self.count]['labels']
            num += 1
            self.count += 1
            if self.count >= len(label):
                np.random.shuffle(label)
                self.count = 0
                self.epoch += 1
        return images, labels


    def next_batches_test(self, label):
        images = np.zeros([self.batch_size, self.image_size, self.image_size, 3])
        labels = np.zeros([self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 5 + self.num_classes])
        num = 0
        while num < self.batch_size:
            imagename = label[self.count_t]['imagename']
            images[num, :, :, :] = self.image_read(imagename)
            labels[num, :, :, :, :] = label[self.count_t]['labels']
            num += 1
            self.count_t += 1
            if self.count_t >= len(label):
                self.count_t = 0
        return images, labels


    def image_read(self, imagename):
        image = cv2.imread(imagename)
        image = cv2.resize(image, (self.image_size, self.image_size))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0 * 2.0 - 1.0
        return image

    def load_map_data(self, model):
        if model == 'train':
            self.data_path = os.path.join(self.data_dir, 'train')
            image_list = glob('{}/*.jpg'.format(self.data_path))
        if model == 'test':
            self.data_path = os.path.join(self.data_dir, 'val')
            image_list = glob('{}/*.jpg'.format(self.data_path))
            print(len(image_list))

        data = []
        for img_path in image_list:
            label, num = self.load_map_labels(img_path)
            if num == 0:
                continue
            data.append({'image_path': img_path, 'label': label})
        return data

    def load_map_labels(self, img_path):
        path_no_suffix = img_path.split('.')[0]
        lab_path = path_no_suffix + '.xml'
        tree = ET.parse(lab_path)
        objects = tree.findall('object')
        objects_num = len(objects)

        label = []
        for object in objects:
            bbox = object.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            y1 = int(bbox.find('ymin').text)
            x2 = int(bbox.find('xmax').text)
            y2 = int(bbox.find('ymax').text)
            class_i = self.class_to_ind[object.find('name').text.lower().strip()]
            if class_i == 0:
                label.append(['face', x1, y1, x2, y2])
            elif class_i == 1:
                label.append(['face_mask', x1, y1, x2, y2])

        return label, objects_num

