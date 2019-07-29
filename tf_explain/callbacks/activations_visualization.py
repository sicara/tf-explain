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
        super(ActivationsVisualizationCallback, self).__init__()
        self.validation_data = validation_data
        self.layers_name = layers_name
        self.output_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        Path.mkdir(Path(self.output_dir), parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        """ Draw activations outputs at each epoch end. """
        explainer = ExtractActivations()
        grid = explainer.explain(self.validation_data, self.model, self.layers_name)

        # Creates a file writer for the log directory.
        file_writer = tf.summary.create_file_writer(str(self.output_dir))

        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image(
                "Activations Visualization",
                np.array([np.expand_dims(grid, axis=-1)]),
                step=epoch,
            )
