from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from tf_explain.utils.display import grid_display


class SmoothGrad:

    """
    Perform SmoothGrad algorithm for a given input

    Paper: [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
    """

    def explain(self, validation_data, model, class_index, num_samples=5, noise=1.0):
        images, _ = validation_data

        noisy_images = SmoothGrad.generate_noisy_images(images, num_samples, noise)

        smoothed_gradients = SmoothGrad.get_averaged_gradients(
            noisy_images, model, class_index, num_samples
        )

        grayscale_gradients = SmoothGrad.transform_to_grayscale(
            smoothed_gradients
        ).numpy()

        grid = grid_display(grayscale_gradients)

        return grid

    @staticmethod
    def generate_noisy_images(images, num_samples, noise):
        repeated_images = np.repeat(images, num_samples, axis=0)
        noise = np.random.normal(0, noise, repeated_images.shape).astype(np.float32)

        return repeated_images + noise

    @staticmethod
    @tf.function
    def transform_to_grayscale(gradients):
        grayscale_grads = tf.reduce_sum(tf.abs(gradients), axis=-1)
        normalized_grads = tf.cast(
            255
            * (grayscale_grads - tf.reduce_min(grayscale_grads))
            / (tf.reduce_max(grayscale_grads) - tf.reduce_min(grayscale_grads)),
            tf.uint8,
        )

        return normalized_grads

    @staticmethod
    @tf.function
    def get_averaged_gradients(noisy_images, model, class_index, num_samples):
        num_classes = model.output.shape[1]

        expected_output = tf.one_hot([class_index] * len(noisy_images), num_classes)

        with tf.GradientTape() as tape:
            inputs = tf.cast(noisy_images, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            loss = tf.keras.losses.categorical_crossentropy(
                expected_output, predictions
            )

        grads = tape.gradient(loss, inputs)

        grads_per_image = tf.reshape(grads, (-1, num_samples, *grads.shape[1:]))
        averaged_grads = tf.reduce_mean(grads_per_image, axis=1)

        return averaged_grads

    def save(self, grid, output_dir, output_name):
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

        cv2.imwrite(str(Path(output_dir) / output_name), grid)
