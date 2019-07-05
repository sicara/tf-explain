import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope='session')
def random_data():
    batch_size = 4
    x = np.random.random((batch_size, 28, 28, 3))
    y = tf.keras.utils.to_categorical(np.random.randint(2, size=batch_size)).astype('uint8')

    return x, y
