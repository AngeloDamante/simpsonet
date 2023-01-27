"""
Train custom model with SGD optimizer and input prepared dataset.
"""
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
from typing import Tuple


def train_model(
        model: tf.keras.Sequential,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray) -> Tuple[tf.keras.Sequential, float, float]:
    """
    Train model with input dataset.

    :param model:
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: compiled model
    :return: accuracy
    :return: log_loss
    """
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=100, batch_size=32)
    target_prob_test = model.predict(x_test)
