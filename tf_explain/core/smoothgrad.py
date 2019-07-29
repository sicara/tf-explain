from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from tf_explain.utils.display import grid_display, heatmap_display


class SmoothGrad:

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(self, validation_data, model, class_index, num_samples=5, noise=1.0):
        images, _ = validation_data

        noisy_images = SmoothGrad.generate_noisy_images(images, num_samples, noise)

        smoothed_gradients = SmoothGrad.get_averaged_gradients(
            noisy_images, model, class_index, num_samples
        ).numpy()

        grid = grid_display(smoothed_gradients)

        return grid

    @staticmethod
    def generate_noisy_images(images, num_samples, noise):
        repeated_images = np.repeat(images, num_samples, axis=0)
        noise = np.random.normal(0, noise, repeated_images.shape).astype(np.float32)

        return repeated_images + noise

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

        grid = np.sum(np.abs(grid), axis=-1)
        grid = ((grid - grid.min()) / (grid.max() - grid.min()) * 255).astype("uint8")

        cv2.imwrite(str(Path(output_dir) / output_name), grid)
