"""
Callback Module for SmoothGrad
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tf_explain.core.smoothgrad import SmoothGrad


class SmoothGradCallback(Callback):

    """
    Perform SmoothGrad algorithm for a given input

    Paper: [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825)
    """

    def __init__(
        self,
        validation_data,
        class_index,
        num_samples=5,
        noise=1.0,
        output_dir=Path("./logs/smoothgrad"),
    ):
        """
        Constructor.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            class_index (int): Index of targeted class
            num_samples (int): Number of noisy samples to generate for each input image
            noise (float): Standard deviation for noise normal distribution
            output_dir (str): Output directory path
        """
        super(SmoothGradCallback, self).__init__()
        self.validation_data = validation_data
        self.class_index = class_index
        self.num_samples = num_samples
        self.noise = noise
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

        self.file_writer = tf.summary.create_file_writer(str(self.output_dir))

    def on_epoch_end(self, epoch, logs=None):
        """
        Draw SmoothGrad outputs at each epoch end to Tensorboard.

        Args:
            epoch (int): Epoch index
            logs (dict): Additional information on epoch
        """
        explainer = SmoothGrad()
        grid = explainer.explain(
            self.validation_data,
            self.model,
            self.class_index,
            self.num_samples,
            self.noise,
        )

        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            tf.summary.image("SmoothGrad", np.expand_dims([grid], axis=-1), step=epoch)
