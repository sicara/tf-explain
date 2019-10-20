"""
Core Module for Gradients*Inputs
"""
import tensorflow as tf

from tf_explain.core.vanilla_gradients import VanillaGradients


class GradientsInputs(VanillaGradients):

    """
    Perform Gradients*Inputs algorithm (gradients ponderated by the input values).
    """

    @staticmethod
    @tf.function
    def compute_gradients(images, model, class_index):
        """
        Compute gradients ponderated by input values for target class.

        Args:
            images (numpy.ndarray): 4D-Tensor of images with shape (batch_size, H, W, 3)
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class

        Returns:
            tf.Tensor: 4D-Tensor
        """
        gradients = VanillaGradients.compute_gradients(images, model, class_index)
        inputs = tf.cast(images, tf.float32)

        return tf.multiply(inputs, gradients)
