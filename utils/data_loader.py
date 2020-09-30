'''
read img and labels
line :
img_path,img_h,img_w,cat_id1,xmin_1,ymin_1,xmax_1,ymax_1,cat_id2,xmin_2,...
'''
import os
import cv2
import numpy as np

class Dataloader(object):

    def __init__(self, height, width, class_num, anchors):
        self.height = height
        self.width = width
        self.class_num = class_num
        self.anchors = anchors

    def read_img(self, img_path):
        '''
        :param img_path:
        :return: BGR image
        '''
        if not os.path.exists(img_path):
            return None
        img = cv2.imread(img_path)
        return img


    def read_label(self, line):
        '''
        :param line: line with box info
            [cat1,xmin1,ymin1,xmax1,ymax1,cat2,xmin2,ymin2,xmax2,ymax2,...]
        :return: label_y1, label_y2, label_y3
        '''
        if not line:
            return None, None, None
        label_y1 = np.zeros(dtype=np.float32, shape=(self.height // 32, self.width // 32, 3, 4+1+self.class_num))
        label_y2 = np.zeros(dtype=np.float32, shape=(self.height // 16, self.width // 16, 3, 4 + 1 + self.class_num))
        label_y3 = np.zeros(dtype=np.float32, shape=(self.height // 8, self.width // 8, 3, 4 + 1 + self.class_num))
        y_true = [label_y1, label_y2, label_y3]
        for i in range(3, len(line), 5):
            box_cat_id = int(line[i])

            # xmin, ymin, xmax, ymax
            box = [line[i + 1], line[i + 2], line[i + 3], line[i + 4]]
            box = np.array(box).astype(np.float32)
            box_wh = box[2:4] - box[0:2]
            box_xy = box[0:2] + box_wh / 2

            # decide which anchor predicts the box
            max_giou = 0
            index = 0
            for i in range(len(self.anchors)):
                min_wh = np.minimum(box_wh, self.anchors[i])
                max_wh = np.maximum(box_wh, self.anchors[i])
                giou = min_wh[0] * min_wh[1] / (max_wh[0] * max_wh[1])
                if giou > max_giou:
                    max_giou = giou
                    index = i

            # match cur box to the correct scale
            anchors_mapping = {0: 8, 1: 16, 2: 32}
            scale_index = index // 3
            anchor_index = index % 3
            x = int(np.floor(box_xy[0] * self.width / anchors_mapping[scale_index]))
            y = int(np.floor(box_xy[1] * self.height / anchors_mapping[scale_index]))
            y_true[scale_index][y, x, anchor_index, 0:4] = np.concatenate((box_xy, box_wh))
            y_true[scale_index][y, x, anchor_index, 4:5] = 1.0
            y_true[scale_index][y, x, anchor_index, 5 + box_cat_id] = 1.0
        return label_y1, label_y2, label_y3


    def get_batch_data(self, batch_line):
        imgs = []
        labels_y1, labels_y2, labels_y3 = [], [], []
        for line in batch_line:
            img_info = line.strip().split(',')
            if len(img_info) < 3:
                raise Exception('[get_batch_line] error: get annotations failed for img: [{}]'.format(img_info[0]))
            img_path, img_height, img_width = img_info[0], img_info[1], img_info[2]
            img = self.read_img(img_path)
            if img is None:
                print("img file: [{}] reading failed. skipped.".format(img_path))

            if len(img_info) % 5 != 0:
                raise Exception('error label for img: [{}]'.format(img_path))
            label_y1, label_y2, label_y3 = self.read_label(img_info[3:])

            imgs.append(img)
            labels_y1.append(label_y1)
            labels_y2.append(label_y2)
            labels_y3.append(label_y3)
        imgs = np.array(imgs)
        labels_y1 = np.asarray(labels_y1)
        labels_y2 = np.asarray(labels_y2)
        labels_y3 = np.asarray(labels_y3)
        return imgs, labels_y1, labels_y2, labels_y3
