import os
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator


class Discriminator:

    ''' Build Discriminator '''
    def __init__(self):

        self.arch = Sequential()

        # Fix input_shape later

        self.arch.add(Conv2D(32, 7, padding='same', strides=2, input_shape=(256, 256, 3)))
        self.arch.add(LeakyReLU(0.2))
        self.arch.add(Dropout(0.1))

        self.arch.add(Conv2D(64, 5, padding='same', strides=1))
        self.arch.add(LeakyReLU(0.2))
        self.arch.add(Dropout(0.1))

        self.arch.add(Conv2D(128, 3, padding='same', strides=1))
        self.arch.add(LeakyReLU(0.2))
        self.arch.add(Dropout(0.1))

        self.arch.add(Conv2D(256, 3, padding='same', strides=1))
        self.arch.add(LeakyReLU(0.2))
        self.arch.add(Dropout(0.1))

        self.arch.add(Flatten())

        # self.arch.add(Dense(128, activation='relu'))
        self.arch.add(Dense(2, activation='sigmoid'))

        adam_lr = 0.0002
        adam_beta_1 = 0.5

        self.arch.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        return

    def __repr__(self):
        return str(self.arch.summary())

    def fit_data(self, train_data, train_labels, test_data, test_labels):
        train_history = self.arch.fit(train_data, train_labels,
                                      epochs=5, steps_per_epoch=1000, validation_steps=200,
                                      validation_data=(test_data, test_labels))
        return train_history

    def fit_gen(self, train_dir, val_dir, num_train, num_val, batch_size, epochs):

        gen = ImageDataGenerator()
        train_generator = gen.flow_from_directory(
            train_dir,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical'
        )
        val_generator = gen.flow_from_directory(
            val_dir,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical'
        )
        train_history = self.arch.fit_generator(
            train_generator,
            steps_per_epoch=(num_train // batch_size),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=(num_val // batch_size)
        )
        
        return train_history

    def save(self, output_dir):
        self.arch.save(os.path.join(output_dir,  'model.h5'))
        return

    def load(self, model_dir):
        self.arch.load(os.path.join(model_dir, 'model.h5'))
        return
