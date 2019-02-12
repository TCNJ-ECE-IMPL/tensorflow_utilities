import os
import sys
import cv2
import json
import numpy
import argparse
import tensorflow

from tensorflow.examples.tutorials.mnist import input_data

def parse_args():
    parser = argparse.ArgumentParser(description='Script to test dataset creation')

    rgroup = parser.add_argument_group('Required Arguments')

    ogroup = parser.add_argument_group('Optional Arguments')

    rgroup.add_argument('--save',
                        help='If FLAG is included script will keep generated data set',
                        required=False,
                        action='store_true')

    rgroup.add_argument('--data_dir',
                        help='Directory to save data set to',
                        required=False,
                        type=str,
                        default=os.environ['DCNN_DATASETS_PATH'])


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    data = input_data.read_data_sets('data/fashion')

    train_images = data.train.images
    train_labels = data.train.labels

    test_images = data.test.images
    test_labels = data.test.labels

    data_dir = sys.argv[1]
    phases = ['train', 'test']

    for phase in phases:
        [os.mkdir(os.path.join(data_dir, 'fashion_mnist/{}/{}'.format(phase,lb))) for lb in set(train_labels)]

        for img, lb in zip(train_images, train_labels):
            im_count = len(os.listdir(
                os.path.join(data_dir, 'fashion_mnist/{}/{}'.format(phase, lb))))
            img_path = os.path.join(data_dir, 'fashion_mnist/{}/{}/{}.png'.format(phase,lb, im_count))
            img = img.reshape((28,28))*255
            cv2.imwrite(img_path, img)
