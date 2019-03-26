import os
import sys
import time
import argparse
import importlib
import numpy as np
import tensorflow as tf

import inspect
import IMPL_Models
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

    rgroup.add_argument('--model',
                        help='Model to use for training',
                        required=True,
                        choices=IMPL_Models.__all__,
                        type=str)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    ds_info = load_data_set_path_dict()[args.dataset]
    dataset = ClassificationDataSet(data_set_dir=ds_info['data_set_dir'])

    print('----- Data Set: \n{}'.format(dataset))

    # Loading images as np arrays
    train_data = np.asarray(np.expand_dims(dataset.train_images, axis=3))
    train_labels = tf.one_hot(
        indices=np.asarray(dataset.train_labels).astype(np.int32),
        depth=dataset.num_classes)

    val_data = np.asarray(np.expand_dims(dataset.test_images, axis=3))
    val_labels = tf.one_hot(
        indices=np.asarray(dataset.test_labels).astype(np.int32),
        depth=dataset.num_classes)

    print('----- Images Loaded: \nTrain: {}\nTest: {}'.format(
        len(train_data), len(val_data)
    ))

    model = IMPL_Models.load_model(args.model)
    print(model)

    results = model.fit_data(train_data, train_labels, val_data, val_labels)
