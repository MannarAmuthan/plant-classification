import pathlib

import tensorflow as tf
from tensorflow.keras import layers


def load_keras_dataset():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    return data_dir


def create_dataset(data_directory: any,image_size: tuple, type: str,batch_size: int,validation_split=0.2):
    dataset=tf.keras.preprocessing.image_dataset_from_directory(
        data_directory,
        validation_split=validation_split,
        subset=type,
        seed=123,
        image_size=image_size,
        batch_size=batch_size)
    return dataset
