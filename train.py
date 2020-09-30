'''
yolov4 - tensorflow1.15
'''


import os
import config
import time
import argparse
import tensorflow as tf
from utils.data_loader import Dataloader
from model.yolov4 import YOLOV4


img_height = int(config.height)
img_width = int(config.width)
batch_size = int(config.batch_size)
class_num = int(config.class_num)
anchors = config.anchorts

train_file = config.train_file
val_file = config.val_file
test_file = config.test_file

def eval():
    pass


# replace this func to which in data_loader
def get_batch_data():
    pass


def read_file(file_path):
    '''
    read train/val/test file and return a list
    :param: str
    :return: list
    '''
    # input data
    if not os.path.exists(file_path):
        raise Exception("data file [{}] not exist! please check your config".format(file_path))
    file_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            file_list.append(line.strip())
    return file_list


def train():
    train_file_list = read_file(train_file)
    dataset = tf.data.Dataset.from_tensor_slices(train_file_list)
    dataset.batch()
    dataset.prefetch(buffer_size=8 * batch_size)
    dataset.map(get_batch_data)


    batch_size = config.batch_size
    batch_norm_params = {}
    image = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3])
    label_19 = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3, 4+1+config.class_num])
    label_38 = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3, 4+1+config.class_num])
    label_76 = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3, 4+1+config.class_num])
    yolo = YOLOV4(class_num=config.class_num,
                    weight_decay=config.weight_decay,
                    batch_norm_params=batch_norm_params)
    feature_19, feature_38, feature_76 = yolo.forward(image)

    pass


def process():
    pass


def main():
    pass


if __name__ == '__main__':
    main()