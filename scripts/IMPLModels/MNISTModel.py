from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class MNISTModel:
    def __init__(self):
        self.model = Sequential()
        # add Convolutional layers
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same',input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Flatten())
        # Densely connected layers
        self.model.add(Dense(128, activation='relu'))
        # output layer
        self.model.add(Dense(10, activation='softmax'))
        # compile with adam optimizer & categorical_crossentropy loss function
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return

    def build_model(self):
        # print(self.model.summary())
        return

    def __str__():
        return self.model.summary()

    def fit_data(self, train_data, train_labels, test_data, test_labels):
        fit = self.model.fit(train_data, train_labels,
                         epochs=2, steps_per_epoch=1000, validation_steps=200,
                         validation_data=(test_data, test_labels))
        return fit
