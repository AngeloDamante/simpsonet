import unittest
import logging

import tensorflow.python.keras

from src.create_dataset import load_dataset, prepare_dataset
from src.create_cnn import train_model, create_cnn
from src.data import configure_logging, k_img_size, k_batch_size

k_dataset_dir = "../dataset"


class TestCreateDataset(unittest.TestCase):
    def test_load_dataset(self):
        configure_logging("log_ut_create_dataset.log", True, log_lvl=logging.DEBUG)
        flag, _, _ = load_dataset(k_dataset_dir)
        self.assertEqual(flag, True)

        flag, _, _ = load_dataset("/dataset")
        self.assertEqual(flag, False)

    def test_prepare_dataset(self):
        _, a, b = load_dataset(k_dataset_dir)
        flag, _ = prepare_dataset(a, b)
        self.assertEqual(flag, True)


class TestTrainCnn(unittest.TestCase):
    cnn_model = None
    x_train = None
    x_test = None
    y_train = None
    y_test = None

    def test_create_cnn(self):
        flag_load_dataset, images, labels = load_dataset(k_dataset_dir)
        self.assertEqual(flag_load_dataset, True)

        flag_prepare_dataset, dataset = prepare_dataset(images, labels)
        self.assertEqual(flag_prepare_dataset, True)

        self.x_train = dataset[0]
        self.x_test = dataset[1]
        self.y_train = dataset[2]
        self.y_test = dataset[3]

        cnn_model = create_cnn(k_img_size)
        self.assertEqual(type(cnn_model), tensorflow.python.keras.Model)

    def test_train_cnn(self):
        compiled_model, history = train_model(self.cnn_model,
                                              self.x_train,
                                              self.x_test,
                                              self.y_train,
                                              self.y_test)

        compiled_model.save("test_cnn.h5")
        with open("test_history.txt", "w+") as file:
            file.write(history)


if __name__ == '__main__':
    unittest.main()
