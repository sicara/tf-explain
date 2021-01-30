"""
Core Module for Vanilla Gradients
"""
import tensorflow as tf
from warnings import warn

from tf_explain.utils.display import grid_display
from tf_explain.utils.image import transform_to_normalized_grayscale
from tf_explain.utils.saver import save_grayscale


_unsupported_architecture_warning = "\
Unsupported model architecture for VanillaGradients. The last two layers of \
the model should be: a layer which computes class scores with no activation, \
followed by an activation layer."

_activation_layer_classes = (
    tf.keras.layers.Activation,
    tf.keras.layers.LeakyReLU,
    tf.keras.layers.PReLU,
    tf.keras.layers.ReLU,
    tf.keras.layers.Softmax,
    tf.keras.layers.ThresholdedReLU
)


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
            model (tf.keras.Model): tf.keras model to inspect. The last two layers of
                the model should be: a layer which computes class scores with no
                activation, followed by an activation layer.
            class_index (int): Index of targeted class

        Returns:
            numpy.ndarray: Grid of all the gradients
        """
        images, _ = validation_data

        score_model = self._get_score_model(model)
        gradients = self.compute_gradients(images, score_model, class_index)

        grayscale_gradients = transform_to_normalized_grayscale(
            tf.abs(gradients)
        ).numpy()

        grid = grid_display(grayscale_gradients)

        return grid

    def _get_score_model(self, model):
        """
        Create a new model that excludes the final Softmax layer from the given model.

        Args:
            model (tf.keras.Model): tf.keras model to base the new model on

        Returns:
            tf.keras.Model: A new model which excludes the last layer
        """
        activation_layer = model.layers[-1]
        if not self._is_activation_layer(activation_layer):
            warn(_unsupported_architecture_warning, stacklevel=2)

        output = activation_layer.input

        return tf.keras.Model(inputs=model.inputs, outputs=output)

    def _is_activation_layer(self, layer):
        """
        Check whether the given layer is an activation layer.

        Args:
            layer (tf.keras.layers.Layer): The layer to check

        Returns:
            Whether the layer is an activation layer
        """
        return isinstance(layer, _activation_layer_classes)

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
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            scores = model(inputs)
            scores_for_class = scores[:, class_index:class_index+1]

        return tape.gradient(scores, inputs)

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Gtid of all the smoothed gradients
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        save_grayscale(grid, output_dir, output_name)
