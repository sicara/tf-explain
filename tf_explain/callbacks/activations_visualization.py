"""
Callback Module for Activations Visualization
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from tf_explain.core.activations import ExtractActivations


class ActivationsVisualizationCallback(Callback):

    """ Draw activations of a specific layer for a given input """

    def __init__(
        self,
        validation_data,
        layers_name,
        output_dir=Path("./logs/activations_visualizations"),
    ):
        """
        Constructor.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            layers_name (List[str]): List of layer names to inspect
            output_dir (str): Output directory path
        """
        super(ActivationsVisualizationCallback, self).__init__()
        self.validation_data = validation_data
        self.layers_name = layers_name
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

        self.file_writer = tf.summary.create_file_writer(str(self.output_dir))

    def on_epoch_end(self, epoch, logs=None):
        """
        Draw activations outputs at each epoch end to Tensorboard.

        Args:
            epoch (int): Epoch index
            logs (dict): Additional information on epoch
        """
        explainer = ExtractActivations()
        grid = explainer.explain(self.validation_data, self.model, self.layers_name)

        # Using the file writer, log the reshaped image.
        with self.file_writer.as_default():
            tf.summary.image(
                "Activations Visualization",
                np.array([np.expand_dims(grid, axis=-1)]),
                step=epoch,
            )
