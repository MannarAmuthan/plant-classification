import numpy as np
import tensorflow as tf
import sys

from tensorflow import keras

from dataset import get_train_validation_of_flower_dataset
from model import CnnModel

img_height = 180
img_width = 180


def get_trained_model(dataset_provider):
    already_trained = False
    dataset_name, class_names, number_of_classes, train_ds, validation_ds = dataset_provider()

    cnn_model = CnnModel(dataset_name, keras.models.load_model(dataset_name)) \
        if already_trained is True \
        else CnnModel.create_cnn_model(dataset_name,number_of_classes, img_height, img_width)

    if already_trained is False:
        cnn_model.fit(train_ds,validation_ds)
        cnn_model.save()

    return (cnn_model,class_names)


if __name__ == "__main__":
    img_to_recognize=sys.argv[1]

    model,class_names=get_trained_model(get_train_validation_of_flower_dataset)

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