import numpy as np
import tensorflow as tf
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

batch_size = 32
img_height = 180
img_width = 180

def get_tf_data_directory():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    return data_dir


def get_dataset_for(data_directory: any, type: str):
    dataset=tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        validation_split=0.2,
        subset=type,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return dataset


def normalize_dataset(dataset):
    normalization_layer=layers.experimental.preprocessing.Rescaling(1. / 255)
    return dataset.map(lambda x, y: (normalization_layer(x), y))

def get_data_augmentation_layer():
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


def get_model():
    model = Sequential([
        get_data_augmentation_layer(),
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def save_model(model,name):
    model.save(name)


if __name__=="__main__":
    img_to_recognize=sys.argv[1]

    already_trained = True

    train_ds = get_dataset_for(get_tf_data_directory(), "training")
    validation_ds = get_dataset_for(get_tf_data_directory(), "validation")
    class_names, num_classes = train_ds.class_names, len(train_ds.class_names)
    normalized_ds = normalize_dataset(train_ds)
    model = keras.models.load_model("flowers_model") if already_trained is True else get_model()

    if already_trained == False:
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=10
        )
        save_model(model, "flowers_model")

    img = keras.preprocessing.image.load_img(
        img_to_recognize,
        target_size=(img_height, img_width)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )