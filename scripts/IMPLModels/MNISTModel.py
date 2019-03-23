from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class MNISTModel:
    def __init__(self):
        self.arch = Sequential()
        # add Convolutional layers
        self.arch.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',input_shape=(28, 28, 1)))
        self.arch.add(MaxPooling2D(pool_size=(2,2)))
        self.arch.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        self.arch.add(MaxPooling2D(pool_size=(2,2)))
        self.arch.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        self.arch.add(MaxPooling2D(pool_size=(2,2)))
        self.arch.add(Flatten())
        # Densely connected layers
        self.arch.add(Dense(128, activation='relu'))
        # output layer
        self.arch.add(Dense(10, activation='softmax'))
        # compile with adam optimizer & categorical_crossentropy loss function
        self.arch.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return

    def __repr__(self):
        return str(self.arch.summary())

    def fit_data(self, train_data, train_labels, test_data, test_labels):
        train_history = self.arch.fit(train_data, train_labels,
                         epochs=2, steps_per_epoch=1000, validation_steps=200,
                         validation_data=(test_data, test_labels))
        return train_history
