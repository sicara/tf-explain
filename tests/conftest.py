from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def random_data():
    batch_size = 4
    x = np.random.random((batch_size, 28, 28, 3))
    y = tf.keras.utils.to_categorical(
        np.random.randint(2, size=batch_size), num_classes=2
    ).astype("uint8")

    return x, y


@pytest.fixture()
def convolutional_model(random_data):
    x, y = random_data
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16,
                (3, 3),
                activation=None,
                name="conv_1",
                input_shape=list(x.shape[1:]),
            ),
            tf.keras.layers.ReLU(name="activation_1"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="categorical_crossentropy")

    return model


@pytest.fixture()
def output_dir():
    return Path(__file__).parent / "tests_outputs"
