

class MNISTModel:
    def __init__(self):
        self.model = Sequential()
        return

    def build_model(self):
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

    def __repr__():
        return self.model.summary()
