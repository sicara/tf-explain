"""
Callback Module for Integrated Gradients
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tf_explain.core.integrated_gradients import IntegratedGradients


class IntegratedGradientsCallback(Callback):

    """
    Perform Integrated Gradients algorithm for a given input

    Paper: [Axiomatic Attribution for Deep Networks](https://arxiv.org/pdf/1703.01365.pdf)
    """

    def __init__(
        self,
        validation_data,
        class_index,
        n_steps=5,
        output_dir=Path("./logs/integrated_gradients"),
    ):
        """
        Constructor.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            class_index (int): Index of targeted class
            n_steps (int): Number of steps in the path
            output_dir (str): Output directory path
        """
        super(IntegratedGradientsCallback, self).__init__()
        self.validation_data = validation_data
        self.class_index = class_index
        self.n_steps = n_steps
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

        self.file_writer = tf.summary.create_file_writer(str(self.output_dir))

    def on_epoch_end(self, epoch, logs=None):
        """
        Draw Integrated Gradients outputs at each epoch end to Tensorboard.

        Args:
            epoch (int): Epoch index
            logs (dict): Additional information on epoch
        """
        explainer = IntegratedGradients()
        grid = explainer.explain(
            self.validation_data, self.model, self.class_index, self.n_steps
        )

        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            tf.summary.image(
                "IntegratedGradients", np.expand_dims([grid], axis=-1), step=epoch
            )
