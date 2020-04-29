import os
import sys
import time
import json
import argparse
import importlib
import numpy as np
from keras import backend as K
import tensorflow as tf

import inspect
import IMPL_Models
from TFUtils.DataSets import ClassificationDataSet
from TFUtils.InOutUtils import load_data_set_info_from_json, load_data_set_info_from_dir

# Handle non-serializable np.float32 when dumping training history to JSON
def float32_default(o):
    if isinstance(o, np.float32):
        return np.float64(o)

def parse_args():

    ds_options = [x for x in os.listdir(os.environ['DCNN_DATASETS_PATH'])]

    parser = argparse.ArgumentParser(description='Train models')

    rgroup = parser.add_argument_group('Required Arguments')

    rgroup.add_argument('--model',
                        help='Model to use for training',
                        required=True,
                        choices=IMPL_Models.__all__,
                        type=str)

    rgroup.add_argument('--epochs',
                        help='Number of epochs to train model for',
                        required=True,
                        type=int)

    rgroup.add_argument('--output_dir',
                        help='Directory to log results to',
                        required=True,
                        type=str)

    ogroup = parser.add_argument_group('Optional Argument')

    ogroup.add_argument('--dataset',
                        help='Data Set to use (options shown to left)',
                        required=False,
                        choices=ds_options,
                        default='',
                        type=str)

    ogroup.add_argument('--dataset_dir',
                        help='Path to existing dataset files',
                        required=False,
                        default='',
                        type=str)

    ogroup.add_argument('--gpu',
                        help='Flag: When input, will use the gpu of the corresponding number entered',
                        required=False,
                        type=str)

    args = parser.parse_args()

    if (args.dataset == args.dataset_dir == ''):
        print('Must specify dataset or dataset_dir')
        exit(0)

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    else:
        print("No GPU specified; using CPU instead")
        tf.config.set_visible_devices([], 'GPU') # Force TensorFlow to use CPU

    if args.dataset:
        ds_info = load_data_set_info_from_json(args.dataset)

    else:
        ds_info = load_data_set_info_from_dir(args.dataset_dir)


    dataset = ClassificationDataSet(data_set_description=ds_info)
    print(dataset)

    print('----- Data Set: \n{}'.format(dataset))

    model = IMPL_Models.load_model(args.model)
    print(model)

    batch_size = 32

    # Adjust batch size to ensure dataset size <= batch size
    for phase in ds_info["data_set_size"]:
        if(ds_info["data_set_size"][phase] < batch_size):
            batch_size = ds_info["data_set_size"][phase]

    results = model.fit_gen(
        train_dir=dataset.train_dir,
        val_dir=dataset.validation_dir,
        num_train=dataset.data_set_size['train'],
        num_val=dataset.data_set_size['validation'],
        epochs=args.epochs,
        batch_size=batch_size)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    model.save(args.output_dir)

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(results.history, f, default=float32_default, sort_keys=True, indent=4)
