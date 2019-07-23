from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from tf_explain.utils.display import heatmap_display


class GradCAM:

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(self, validation_data, model, layer_name, class_index):
        images, _ = validation_data

        output, guided_grads = GradCAM.get_gradients_and_filters(
            model, images, layer_name, class_index
        )

        cam = GradCAM.generate_ponderated_output(output, guided_grads)

        heatmap = heatmap_display(cam, images[0])

        return heatmap

    @staticmethod
    def get_gradients_and_filters(model, images, layer_name, class_index):
        """
        Generate guided gradients and convolutional outputs with an inference

        Args:
            model:
            images:
            layer_name:
            class_index:

        Returns:

        """
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            inputs = np.array([images[0]]).astype("float32")
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, class_index]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        guided_grads = (
            tf.cast(output > 0, "float32") * tf.cast(grads > 0, "float32") * grads
        )

        return output, guided_grads

    @staticmethod
    def generate_ponderated_output(output, grads):
        """
        Apply Grad CAM algorithm scheme.

        Inputs are the convolutional outputs (shape WxHxN) and gradients (shape WxHxN).
        From there:
          - we compute the spatial average of the gradients
          - we build a ponderated sum of the convolutional outputs based on those averaged weights

        Args:
            output:
            grads:

        Returns:

        """
        weights = tf.reduce_mean(grads, axis=(0, 1))

        cam = np.ones(output.shape[0:2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        return cam.numpy()

    def save(self, grid, output_dir, output_name):
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

        grid_as_image = Image.fromarray((np.clip(grid, 0, 1) * 255).astype("uint8"))
        grid_as_image.save(Path(output_dir) / output_name)
