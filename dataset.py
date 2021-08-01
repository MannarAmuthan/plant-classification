import pathlib

import tensorflow as tf
from tensorflow.keras import layers


batch_size = 32
img_height = 180
img_width = 180
flower_dataset_url="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_MODEL_NAME = "flowers_model"


def get_train_validation_of_flower_dataset():
    train_ds = create_dataset(get_downloaded_keras_dataset_path(flower_dataset_url,'flower_photos'), (img_height, img_width),
                              "training", batch_size)
    validation_ds = create_dataset(get_downloaded_keras_dataset_path(flower_dataset_url,'flower_photos'), (img_height, img_width),
                                   "validation", batch_size)
    class_names, number_of_classes = train_ds.class_names, len(train_ds.class_names)
    return FLOWERS_MODEL_NAME , class_names, number_of_classes, train_ds, validation_ds


def get_downloaded_keras_dataset_path(dataset_url: str,folder_name: str):
    data_dir = tf.keras.utils.get_file(folder_name, origin=dataset_url, untar=True)
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
