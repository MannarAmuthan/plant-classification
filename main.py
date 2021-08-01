import numpy as np
import tensorflow as tf
import sys

from tensorflow import keras

from dataset import get_downloaded_keras_dataset_path, create_dataset
from model import create_cnn_model, save_model

batch_size = 32
img_height = 180
img_width = 180
flower_dataset_url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


def get_trained_model():
    already_trained = True
    train_ds = create_dataset(get_downloaded_keras_dataset_path(flower_dataset_url), (img_height, img_width), "training", batch_size)
    validation_ds = create_dataset(get_downloaded_keras_dataset_path(flower_dataset_url), (img_height, img_width), "validation", batch_size)
    class_names, number_of_classes = train_ds.class_names, len(train_ds.class_names)
    model = keras.models.load_model("flowers_model") if \
        already_trained is True else create_cnn_model(number_of_classes,img_height,img_width)

    if already_trained is False:
        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=10
        )
        save_model(model, "flowers_model")

    return (model,class_names)


if __name__=="__main__":
    img_to_recognize=sys.argv[1]

    model,class_names=get_trained_model()

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