from dataset import create_dataset, get_downloaded_keras_dataset_path


class FlowersDataset:

    def __init__(self):
        self.class_names=['Hibiscus', 'Jasmine', 'Lotus', 'Roses', 'Sunflower', 'Tuberose']
        self.number_of_classes=len(self.class_names)
        self.MODEL_NAME = "flowers_model_two"
        self.img_height = 180
        self.img_width = 180

    def load(self):
        self.batch_size = 32
        self.train_ds = create_dataset("/Users/amuthanmannan/Downloads/Flowers299/data/train", (self.img_height, self.img_width),
                                       "training", self.batch_size)
        self.validation_ds = create_dataset("/Users/amuthanmannan/Downloads/Flowers299/data/train",
                                            (self.img_height, self.img_width),
                                            "validation", self.batch_size)
        self.class_names = self.train_ds.class_names
        self.number_of_classes = len(self.class_names)
        self.MODEL_NAME = "flowers_model_two"
