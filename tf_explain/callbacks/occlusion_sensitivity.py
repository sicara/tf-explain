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
        patch_size,
        class_index,
        output_dir=Path("./logs/occlusion_sensitivity"),
    ):
        super(OcclusionSensitivityCallback, self).__init__()
        self.validation_data = validation_data
        self.patch_size = patch_size
        self.class_index = class_index
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(self.output_dir, parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        explainer = OcclusionSensitivity()
        grid = explainer.explain(
            self.validation_data, self.model, self.class_index, self.patch_size
        )

        # Creates a file writer for the log directory.
        file_writer = tf.summary.create_file_writer(str(self.output_dir))

        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image("Occlusion Sensitivity", np.array([grid]), step=epoch)
