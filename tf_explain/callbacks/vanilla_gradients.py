"""
Callback Module for Vanilla Gradients
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tf_explain.core.vanilla_gradients import VanillaGradients


class VanillaGradientsCallback(Callback):

    """
    Perform gradients backpropagation for a given input

    Paper: [Deep Inside Convolutional Networks: Visualising Image Classification
        Models and Saliency Maps](https://arxiv.org/abs/1312.6034)
    """

    explainer = VanillaGradients()
    default_output_subdir = "vanilla_gradients"

    def __init__(self, validation_data, class_index, output_dir=None):
        """
        Constructor.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            class_index (int): Index of targeted class
            num_samples (int): Number of noisy samples to generate for each input image
        """
        super(VanillaGradientsCallback, self).__init__()
        self.validation_data = validation_data
        self.class_index = class_index
        if not output_dir:
            output_dir = Path("./logs") / self.default_output_subdir
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

        self.file_writer = tf.summary.create_file_writer(str(self.output_dir))

    def on_epoch_end(self, epoch, logs=None):
        """
        Draw VanillaGradients outputs at each epoch end to Tensorboard.

        Args:
            epoch (int): Epoch index
            logs (dict): Additional information on epoch
        """
        grid = self.explainer.explain(
            self.validation_data, self.model, self.class_index
        )

        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            tf.summary.image(
                self.explainer.__class__.__name__,
                np.expand_dims([grid], axis=-1),
                step=epoch,
            )
