import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import Callback

from visualize.utils.display import filter_display


class ActivationsVisualizationCallback(Callback):

    """ Draw activations of a specific layer for a given input """

    def __init__(self, validation_data, layers_name, output_dir=Path('./logs/activations_visualizations')):
        super(ActivationsVisualizationCallback, self).__init__()
        self.validation_data = validation_data
        self.layers_name = layers_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        outputs = [
            layer.output for layer in self.model.layers
            if layer.name in self.layers_name
        ]

        activations_model = tf.keras.models.Model(self.model.inputs, outputs=outputs)
        activations_model.compile(optimizer='adam', loss='categorical_crossentropy')

        predictions = activations_model.predict(self.validation_data[0])
        grid = filter_display(predictions)

        im = Image.fromarray((np.clip(grid, 0, 1)*255).astype('uint8'))
        im.save(Path(self.output_dir) / f'{epoch}.png')
