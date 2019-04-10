import os
from keras.models import Model, load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras_preprocessing.image import ImageDataGenerator
#from ClassificationModel import ClassificationModel

class ResNetModel:#(ClassificationModel):
    def __init__(self):
        #super(CardModel, self).__init___(model_name='CardModel')
        self.num_classes = 2
        self.build_model()

        return

    def build_model(self):
        # Initializing the model with random wights
        self.arch = ResNet50(weights=None, include_top=True, classes=self.num_classes)

        # Compiling model with optimization of RSM and cross entropy loss
        self.arch.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        return

    def train(self, epochs, train_images, train_labels, val_data, batch_size, initial_epoch=None):
        history = self.arch.fit_generator(
            train_generator,
            epochs=epochs,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size)
        return history

    def fit_gen(self, train_dir, val_dir, num_train, num_val, batch_size, epochs):

        gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_generator = gen.flow_from_directory(
            train_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical'
        )
        val_generator = gen.flow_from_directory(
            val_dir,
            target_size=(224, 224),
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
        model_path = os.path.join(output_dir, 'model.h5')
        self.arch.save(model_path)
        return

    def load(self, input_path):
        model_path = os.path.join(output_dir, 'model.h5')
        self.arch = load_model(model_path)
        return

    def __repr__(self):
        return str(self.arch.summary())