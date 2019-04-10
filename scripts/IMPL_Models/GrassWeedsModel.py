import os
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator

class GrassWeedsModel():
	def __init__(self):
		return

	def build_model(self):
		return

    def __repr__(self):
    	return ''

    def fit_data(self, train_data, train_labels, test_data, test_labels):
        return 

    def fit_gen(self, train_dir, val_dir, num_train, num_val, batch_size, epochs):
        return

    def save(self, output_dir):
        return

    def load(self, model_dir):
        return