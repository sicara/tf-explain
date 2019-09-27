"""
Core Module for Integrated Gradients Algorithm
"""
import numpy as np
import tensorflow as tf

from tf_explain.utils.display import grid_display
from tf_explain.utils.image import transform_to_normalized_grayscale
from tf_explain.utils.saver import save_grayscale


class IntegratedGradients:

    """
    Perform Integrated Gradients algorithm for a given input

    Paper: [Axiomatic Attribution for Deep Networks](https://arxiv.org/pdf/1703.01365.pdf)
    """

    def explain(self, validation_data, model, class_index, n_steps=10):
        """
        Compute Integrated Gradients for a specific class index

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            n_steps (int): Number of steps in the path

        Returns:
            np.ndarray: Grid of all the integrated gradients
        """
        images, _ = validation_data

        interpolated_images = IntegratedGradients.generate_interpolations(
            np.array(images), n_steps
        )

        integrated_gradients = IntegratedGradients.get_integrated_gradients(
            interpolated_images, model, class_index, n_steps
        )

        grayscale_integrated_gradients = transform_to_normalized_grayscale(
            tf.abs(integrated_gradients)
        ).numpy()

        grid = grid_display(grayscale_integrated_gradients)

        return grid

    @staticmethod
    @tf.function
    def get_integrated_gradients(interpolated_images, model, class_index, n_steps):
        """
        Perform backpropagation to compute integrated gradients.

        Args:
            interpolated_images (numpy.ndarray): 4D-Tensor of shape (N * n_steps, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            n_steps (int): Number of steps in the path

        Returns:
            tf.Tensor: 4D-Tensor of shape (N, H, W, 3) with integrated gradients
        """
        with tf.GradientTape() as tape:
            inputs = tf.cast(interpolated_images, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, inputs)
        grads_per_image = tf.reshape(grads, (-1, n_steps, *grads.shape[1:]))

        integrated_gradients = tf.reduce_mean(grads_per_image, axis=1)

        return integrated_gradients

    @staticmethod
    def generate_interpolations(images, n_steps):
        """
        Generate interpolation paths for batch of images.

        Args:
            images (numpy.ndarray): 4D-Tensor of images with shape (N, H, W, 3)
            n_steps (int): Number of steps in the path

        Returns:
            numpy.ndarray: Interpolation paths for each image with shape (N * n_steps, H, W, 3)
        """
        baseline = np.zeros(images.shape[1:])

        return np.concatenate(
            [
                IntegratedGradients.generate_linear_path(baseline, image, n_steps)
                for image in images
            ]
        )

    @staticmethod
    def generate_linear_path(baseline, target, n_steps):
        """
        Generate the interpolation path between the baseline image and the target image.

        Args:
            baseline (numpy.ndarray): Reference image
            target (numpy.ndarray): Target image
            n_steps (int): Number of steps in the path

        Returns:
            List(np.ndarray): List of images for each step
        """
        return [
            baseline + (target - baseline) * index / (n_steps - 1)
            for index in range(n_steps)
        ]

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all the heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_grayscale(grid, output_dir, output_name)
