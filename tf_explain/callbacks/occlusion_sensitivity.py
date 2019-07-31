"""
Callback Module for Occlusion Sensitivity
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity


class OcclusionSensitivityCallback(Callback):
    def __init__(
        self,
        validation_data,
        class_index,
        patch_size,
        output_dir=Path("./logs/occlusion_sensitivity"),
    ):
        """
        Constructor.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image
            output_dir (str): Output directory path
        """
        super(OcclusionSensitivityCallback, self).__init__()
        self.validation_data = validation_data
        self.class_index = class_index
        self.patch_size = patch_size
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(self.output_dir, parents=True, exist_ok=True)

        self.file_writer = tf.summary.create_file_writer(str(self.output_dir))

    def on_epoch_end(self, epoch, logs=None):
        """
        Draw Occlusion Sensitivity outputs at each epoch end to Tensorboard.

        Args:
            epoch (int): Epoch index
            logs (dict): Additional information on epoch
        """
        explainer = OcclusionSensitivity()
        grid = explainer.explain(
            self.validation_data, self.model, self.class_index, self.patch_size
        )

        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            tf.summary.image("Occlusion Sensitivity", np.array([grid]), step=epoch)
