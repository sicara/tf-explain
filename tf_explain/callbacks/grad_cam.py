"""
Callback Module for Grad CAM
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tf_explain.core.grad_cam import GradCAM


class GradCAMCallback(Callback):

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def __init__(
        self,
        validation_data,
        class_index,
        layer_name=None,
        output_dir=Path("./logs/grad_cam"),
    ):
        """
        Constructor.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            class_index (int): Index of targeted class
            layer_name (str): Targeted layer for GradCAM
            output_dir (str): Output directory path
        """
        super(GradCAMCallback, self).__init__()
        self.validation_data = validation_data
        self.layer_name = layer_name
        self.class_index = class_index
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

        self.file_writer = tf.summary.create_file_writer(str(self.output_dir))

    def on_epoch_end(self, epoch, logs=None):
        """
        Draw GradCAM outputs at each epoch end to Tensorboard.

        Args:
            epoch (int): Epoch index
            logs (dict): Additional information on epoch
        """
        explainer = GradCAM()
        heatmap = explainer.explain(
            self.validation_data,
            self.model,
            class_index=self.class_index,
            layer_name=self.layer_name,
        )

        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            tf.summary.image("Grad CAM", np.array([heatmap]), step=epoch)
