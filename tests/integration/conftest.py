import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(scope="session")
def num_classes():
    return 10


@pytest.fixture(scope="session")
def mnist_dataset(num_classes):
    # Load dataset
    dataset = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()

    # Convert from (28, 28) images to (28, 28, 1)
    train_images = train_images[..., tf.newaxis].astype("float32")
    test_images = test_images[..., tf.newaxis].astype("float32")

    # One hot encore labels 0, 1, .., 9 to [0, 0, .., 1, 0, 0]
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

    return (
        train_images[0:500],
        train_labels[0:500],
        test_images[0:300],
        test_labels[0:300],
    )


@pytest.fixture(scope="session")
def validation_dataset(mnist_dataset, num_classes):
    train_images, train_labels, test_images, test_labels = mnist_dataset
    TARGET_CLASS = np.random.choice(num_classes, 1)[0]
    ONE_HOT_TARGET_CLASS = np.array(
        [int(el == TARGET_CLASS) for el in range(num_classes)]
    )

    validation_target_class = (
        np.array(
            [
                el
                for el, label in zip(test_images, test_labels)
                if np.all(label == ONE_HOT_TARGET_CLASS)
            ]
        ),
        None,
    )

    return validation_target_class, TARGET_CLASS
