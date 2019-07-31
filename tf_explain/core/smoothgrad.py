"""
Core Module for SmoothGrad Algorithm
"""
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
        """
        Compute SmoothGrad for a specific class index

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            num_samples (int): Number of noisy samples to generate for each input image
            noise (float): Standard deviation for noise normal distribution

        Returns:
            np.ndarray: Grid of all the smoothed gradients
        """
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
        """
        Generate num_samples noisy images with std noise for each image.

        Args:
            images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
            num_samples (int): Number of noisy samples to generate for each input image
            noise (float): Standard deviation for noise normal distribution

        Returns:
            np.ndarray: 4D-Tensor of noisy images with shape (batch_size*num_samples, H, W, 3)
        """
        repeated_images = np.repeat(images, num_samples, axis=0)
        noise = np.random.normal(0, noise, repeated_images.shape).astype(np.float32)

        return repeated_images + noise

    @staticmethod
    @tf.function
    def transform_to_grayscale(gradients):
        """
        Transform gradients over RGB axis to grayscale.

        Args:
            gradients (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, 3)

        Returns:
            tf.Tensor: 4D-Tensor of grayscale gradients, with shape (batch_size, H, W, 1)
        """
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
        """
        Compute average of gradients for target class.

        Args:
            noisy_images (tf.Tensor): 4D-Tensor of noisy images with shape (batch_size*num_samples, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            num_samples (int): Number of noisy samples to generate for each input image

        Returns:
            tf.Tensor: 4D-Tensor with smoothed gradients, with shape (batch_size, H, W, 1)
        """
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
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Gtid of all the smoothed gradients
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

        cv2.imwrite(str(Path(output_dir) / output_name), grid)
