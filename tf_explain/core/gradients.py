"""
Core Module for Vanilla Gradients
"""
import tensorflow as tf

from tf_explain.utils.display import grid_display
from tf_explain.utils.image import transform_to_normalized_grayscale
from tf_explain.utils.saver import save_grayscale


class VanillaGradients:

    """
    Perform gradients backpropagation for a given input

    Paper: [Deep Inside Convolutional Networks: Visualising Image Classification
        Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
    """

    def explain(self, validation_data, model, class_index):
        """
        Perform gradients backpropagation for a given input

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class

        Returns:
            numpy.ndarray: Grid of all the gradients
        """
        images, _ = validation_data

        gradients = self.compute_gradients(images, model, class_index)

        grayscale_gradients = transform_to_normalized_grayscale(
            tf.abs(gradients)
        ).numpy()

        grid = grid_display(grayscale_gradients)

        return grid

    @staticmethod
    @tf.function
    def compute_gradients(images, model, class_index):
        """
        Compute gradients for target class.

        Args:
            images (numpy.ndarray): 4D-Tensor of images with shape (batch_size, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class

        Returns:
            tf.Tensor: 4D-Tensor
        """
        num_classes = model.output.shape[1]

        expected_output = tf.one_hot([class_index] * images.shape[0], num_classes)

        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            predictions = model(inputs)
            loss = tf.keras.losses.categorical_crossentropy(
                expected_output, predictions
            )

        return tape.gradient(loss, inputs)

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Gtid of all the smoothed gradients
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_grayscale(grid, output_dir, output_name)
