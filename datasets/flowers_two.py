

"/Users/amuthanmannan/Downloads/Flowers299/data"

from dataset import create_dataset, get_downloaded_keras_dataset_path

batch_size = 32
img_height = 180
img_width = 180

MODEL_NAME = "flowers_model_two"


def get_train_validation_of_flower_dataset_two():
    train_ds = create_dataset("/Users/amuthanmannan/Downloads/Flowers299/data/train", (img_height, img_width),
                              "training", batch_size)
    validation_ds = create_dataset("/Users/amuthanmannan/Downloads/Flowers299/data/train", (img_height, img_width),
                                   "validation", batch_size)
    class_names, number_of_classes = train_ds.class_names, len(train_ds.class_names)
    return MODEL_NAME , class_names, number_of_classes, train_ds, validation_ds, img_height , img_width