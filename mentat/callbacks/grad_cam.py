import os
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from mentat.utils.display import heatmap_display


class GradCAMCallback(Callback):

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def __init__(
        self,
        validation_data,
        layer_name,
        class_index,
        output_dir=Path("./logs/grad_cam"),
    ):
        super(GradCAMCallback, self).__init__()
        self.validation_data = validation_data
        self.layer_name = layer_name
        self.class_index = class_index
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """ Draw activations outputs at each epoch end. """
        images, _ = self.validation_data

        output, guided_grads = GradCAMCallback.get_gradients_and_filters(
            self.model, images, self.layer_name, self.class_index
        )

        cam = GradCAMCallback.generate_ponderated_output(output, guided_grads)

        heatmap = heatmap_display(cam.numpy(), images[0])

        cv2.imwrite(str(Path(self.output_dir) / f"{epoch}.png"), heatmap)

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

        return cam

    @staticmethod
    def generate_grad_cam_heatmap(cam, image, target_shape=(224, 224)):
        cam = cv2.resize(cam.numpy(), target_shape)
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

        output_image = cv2.addWeighted(
            cv2.cvtColor(image.astype("uint8"), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0
        )

        return output_image
