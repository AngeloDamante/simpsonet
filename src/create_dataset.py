"""
Create Dataset to training and evaluation a Convolutional NN.
"""
from src.data import characters, k_img_size

# math and vision libs
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# utils
import os
import logging
from typing import Tuple


def load_dataset(path: str, img_size: tuple = k_img_size) -> Tuple[bool, np.ndarray, np.ndarray]:
    """
    To load dataset from desired path.

    :param path: absolute or relative path of dataset folder.
    :param img_size: (width, height)
    :return: flag(bool) to check correctness
    :return: images(ndarray)
    :return: labels(ndarray)
    """
    # preconditions
    if not os.path.isdir(path):
        logging.error("wrong directory")
        return False, np.zeros(1), np.zeros(1)
    if len(os.listdir(path)) != len(characters):
        logging.error("all characters must be detected")
        return False, np.zeros(1), np.zeros(1)

    # create dataset
    pictures = []
    labels = []
    for i, char in characters.items():
        for img in os.listdir(f'{path}/{char}'):
            new_img = cv2.resize(cv2.imread(f'{path}/{char}/{img}'), img_size)
            pictures.append(new_img)
            labels.append(i)
    return True, np.array(pictures), np.array(labels)


def prepare_dataset(images: np.ndarray, labels: np.ndarray) -> Tuple[bool, list]:
    """
    Prepare dataset to fit CNN with one-hot encoding technique.

    :param images
    :param labels
    :return: flag(bool) to verify correctness
    :return: list(x_train, x_test, y_train, y_test)
    """
    # preconditions
    if images.shape[0] != labels.shape[0]:
        return False, []

    # split dataset in One Hot Encoding Format
    labels_ohe = to_categorical(labels, len(characters))
    x_train, x_test, y_train, y_test = train_test_split(images, labels_ohe, test_size=0.13)
    return True, [x_train, x_test, y_train, y_test]
