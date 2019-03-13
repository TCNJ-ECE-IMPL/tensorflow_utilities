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

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
    model = Sequential()
    # add Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    # output layer
    model.add(Dense(10, activation='softmax'))
    # compile with adam optimizer & categorical_crossentropy loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    args = parse_args()

    ds_info = load_data_set_path_dict()[args.dataset]
    dataset = ClassificationDataSet(data_set_dir=ds_info['data_set_dir'])

    print('----- Data Set: \n{}'.format(dataset))

    # Loading images as np arrays
    train_data = np.asarray(np.expand_dims(dataset.train_images, axis=3))
    train_labels = tf.one_hot(
        indices=np.asarray(dataset.train_labels).astype(np.int32),
        depth=10)

    eval_data = np.asarray(np.expand_dims(dataset.test_images, axis=3))
    eval_labels = tf.one_hot(
        indices=np.asarray(dataset.test_labels).astype(np.int32),
        depth=10)

    print('----- Images Loaded: \nTrain: {}\nTest: {}'.format(
        train_data.shape, len(eval_data)
    ))

    model = build_model()
    print(model.summary())

    results = model.fit(train_data, train_labels,
                        epochs=2, steps_per_epoch=1000, validation_steps=200,
                        validation_data=(eval_data, eval_labels))
