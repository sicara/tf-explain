from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from tf_explain.utils.display import grid_display, heatmap_display


class GradCAM:

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def explain(self, validation_data, model, layer_name, class_index):
        images, _ = validation_data

        outputs, guided_grads = GradCAM.get_gradients_and_filters(
            model, images, layer_name, class_index
        )

        cams = GradCAM.generate_ponderated_output(outputs, guided_grads)

        heatmaps = np.array([heatmap_display(cam, image) for cam, image in zip(cams, images)])

        grid = grid_display(heatmaps)

        return grid

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
            inputs = np.array(images).astype("float32")
            conv_outputs, predictions = grad_model(inputs)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)

        guided_grads = (
            tf.cast(conv_outputs > 0, "float32") * tf.cast(grads > 0, "float32") * grads
        )

        return conv_outputs, guided_grads

    @staticmethod
    def generate_ponderated_output(outputs, grads):
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

        maps = []
        for output, grad in zip(outputs, grads):
            weights = tf.reduce_mean(grad, axis=(0, 1))
            cam = np.ones(output.shape[0:2], dtype=np.float32)

            for i, w in enumerate(weights):
                cam += w * output[:, :, i]

            maps.append(cam.numpy())

        return maps

    def save(self, grid, output_dir, output_name):
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

        grid_as_image = Image.fromarray((np.clip(grid, 0, 1) * 255).astype("uint8"))
        grid_as_image.save(Path(output_dir) / output_name)
