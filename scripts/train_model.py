import os
import sys
import time
import argparse
import importlib
import numpy as np
from keras import backend as K
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

    ogroup = parser.add_argument_group('Optional Argument')

    ogroup.add_argument('--gpu',
                        help='Flag: When set GPUs will be used for accelerated training')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        with K.tf.device('/gpu:1'):
            print('Using GPU')
            config = tf.ConfigProto(intra_op_parallelism_threads=4,\
                inter_op_parallelism_threads=4, allow_soft_placement=True,\
                device_count = {'CPU' : 1, 'GPU' : 1})
            session = tf.Session(config=config)
            K.set_session(session)

    ds_info = load_data_set_path_dict()[args.dataset]

    dataset = ClassificationDataSet(data_set_dir=ds_info['data_set_dir'])

    print('----- Data Set: \n{}'.format(dataset))

    model = IMPL_Models.load_model(args.model)

    results = model.fit_gen(dataset.train_dir, dataset.validation_dir, 2)
