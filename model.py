import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import Sequential


class CnnModel:

    @classmethod
    def create_cnn_model(cls,model_name: str,number_of_classes:int,input_image_height,input_image_width,channels=3):
        model = Sequential([
            get_data_augmentation_layer(input_image_height,input_image_width),
            layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(input_image_height, input_image_width, channels)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(number_of_classes)
        ])

        model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

        return CnnModel(model_name,model)

    def __init__(self,name: str,model):
        self.name=name
        self.model=model

    def save(self):
        self.model.save(self.name)

    def fit(self,training_dataset,validation_ds,epochs=10):
        self.model.fit(
            training_dataset,
            validation_data=validation_ds,
            epochs=epochs
        )

    def predict(self,img_array):
        return self.model.predict(img_array)


def get_data_augmentation_layer(img_height,img_width):
    return keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )