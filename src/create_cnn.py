"""
Create CNN with 6 convolutional layers.
"""
import tensorflow.python.keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import accuracy_score, log_loss
from src.data import characters, k_batch_size
import tensorflow as tf
import numpy as np
from keras.optimizers import SGD
from typing import Tuple


def create_cnn(input_shape) -> tensorflow.python.keras.Model:
    """
    Create a CNN with 6 convolutional layers.

    :param input_shape: for first layer
    :return: model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(characters), activation='softmax'))
    return model


def train_model(
        model: tf.keras.Sequential,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray) -> Tuple[tf.keras.Sequential, dict]:
    """
    Train custom model with SGD optimizer and input prepared dataset.

    :param model:
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return: compiled model
    :return: history dict about tmodel
    :return: metrics for test set (accuracy, log_loss)
    """
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=k_batch_size,
        steps_per_epoch=x_train.shape[0] // k_batch_size,
        validation_data=(x_test, y_test)
    )
    return model, history
