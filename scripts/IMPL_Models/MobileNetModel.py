from keras.models import Model, load_model
from keras.applications.mobilenet import MobileNet, preprocess_input

from ClassificationModel import ClassificationModel

class MobileNetModel(ClassificationModel):
    def __init__(self):
        super(CardModel, self).__init___(model_name='CardModel')
        self.num_classes = 13
        self.build_model()

        return

    def build_model(self):
        # Initializing the model with random wights
        self.model = MobileNet(weights=None, include_top=True, classes=self.num_classes)

        print(self.model.summary())

        # Compiling model with optimization of RSM and cross entropy loss
        self.model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return

    def train(self, epochs, train_images, train_labels, val_data, batch_size, initial_epoch=None):
        history = self.model.fit_generator(
            train_generator,
            epochs=epochs,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size)
        return history

    def save_model(self, model_path):
        self.model.save(model_path)
        return

    def load_model(self, model_path):
        self.model = load_model(model_path)
        return
