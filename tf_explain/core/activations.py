from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from tf_explain.utils.display import filter_display


class ExtractActivations:

    """ Draw activations of a specific layer for a given input """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def explain(self, validation_data, model, layers_name):
        activations_model = self.generate_activations_graph(model, layers_name)

        predictions = activations_model.predict(
            validation_data[0], batch_size=self.batch_size
        )
        grid = filter_display(predictions)

        grid = (grid - grid.min()) / (grid.max() - grid.min())

        return (np.clip(grid, 0, 1) * 255).astype("uint8")

    @staticmethod
    def generate_activations_graph(model, layers_name):
        outputs = [layer.output for layer in model.layers if layer.name in layers_name]
        activations_model = tf.keras.models.Model(model.inputs, outputs=outputs)
        activations_model.compile(optimizer="sgd", loss="categorical_crossentropy")

        return activations_model

    def save(self, grid, output_dir, output_name):
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

        cv2.imwrite(
            str(Path(output_dir) / output_name),
            grid,
        )
