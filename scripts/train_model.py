import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf

from TFUtils.DataSets import ClassificationDataSet
from TFUtils.InOutUtils import load_data_set_path_dict

def parse_args():

    ds_options = [x for x in os.listdir(os.environ['DCNN_DATASETS_PATH'])]

    parser = argparse.ArgumentParser(description='Train models')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--dataset',
                        help='Data Set to use (options shown to left)',
                        required=True,
                        choices=ds_options,
                        type=str)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    ds_info = load_data_set_path_dict()[args.dataset]
    dataset = ClassificationDataSet(data_set_dir=ds_info['data_set_dir'])

    print('----- Data Set: \n{}'.format(dataset))

    # Loading images as np arrays
    train_data = np.asarray(dataset.train_images)
    train_labels = tf.one_hot(
        indices=np.asarray(dataset.train_labels).astype(np.int32),
        depth=10)

    eval_data = np.asarray(dataset.test_images)
    eval_labels = tf.one_hot(
        indices=np.asarray(dataset.test_labels).astype(np.int32),
        depth=10)

    print('----- Images Loaded: \nTrain: {}\nTest: {}'.format(
        len(train_data), len(eval_data)
    ))
