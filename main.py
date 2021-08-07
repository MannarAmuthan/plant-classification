import numpy as np
import tensorflow as tf
import sys

from tensorflow import keras

from datasets.flowers_two import FlowersDataset
from model import CnnModel


def get_trained_model(dataset: FlowersDataset):
    already_trained = True

    img_height = dataset.img_height
    img_width = dataset.img_width
    dataset_name = dataset.MODEL_NAME

    if already_trained:
        class_names=dataset.class_names
        cnn_model = CnnModel(dataset_name, keras.models.load_model(dataset_name))
    else:
        dataset.load()
        class_names = dataset.class_names
        number_of_classes = dataset.number_of_classes
        train_ds=dataset.train_ds
        validation_ds=dataset.validation_ds
        cnn_model=CnnModel.create_cnn_model(dataset_name, number_of_classes, img_height, img_width)
        cnn_model.fit(train_ds,validation_ds)
        cnn_model.save()

    return cnn_model,class_names, img_height, img_width


def predict(img):
    model, class_names, img_height, img_width = get_trained_model(FlowersDataset())
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    return predicted_class,score

if __name__ == "__main__":
    img_to_recognize=sys.argv[1]

    img = keras.preprocessing.image.load_img(
        img_to_recognize,
        target_size=(180, 180)
    )

    (predicted_class , score)=predict(img)

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(predicted_class, 100 * np.max(score))
    )