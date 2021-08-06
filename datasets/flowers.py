from dataset import create_dataset, get_downloaded_keras_dataset_path

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
    return FLOWERS_MODEL_NAME , class_names, number_of_classes, train_ds, validation_ds, img_height , img_width