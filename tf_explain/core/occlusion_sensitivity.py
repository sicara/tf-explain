"""
Core Module for Occlusion Sensitivity
"""
import math
from pathlib import Path

import cv2
import numpy as np

from tf_explain.utils.display import grid_display, heatmap_display
from tf_explain.utils.image import apply_grey_patch


class OcclusionSensitivity:

    """
    Perform Occlusion Sensitivity for a given input
    """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def explain(self, validation_data, model, class_index, patch_size):
        """
        Compute Occlusion Sensitivity maps for a specific class index.

        Args:
            validation_data (Tuple[np.ndarray, Optional[np.ndarray]]): Validation data
                to perform the method on. Tuple containing (x, y).
            model (tf.keras.Model): tf.keras model to inspect
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image

        Returns:
            np.ndarray: Grid of all the sensitivity maps with shape (batch_size, H, W, 3)
        """
        images, _ = validation_data
        sensitivity_maps = np.array(
            [
                self.get_sensitivity_map(model, image, class_index, patch_size)
                for image in images
            ]
        )

        heatmaps = np.array(
            [
                heatmap_display(heatmap, image)
                for heatmap, image in zip(sensitivity_maps, images)
            ]
        )

        grid = grid_display(heatmaps)

        return grid

    def get_sensitivity_map(self, model, image, class_index, patch_size):
        """
        Compute sensitivity map on a given image for a specific class index.

        Args:
            model (tf.keras.Model): tf.keras model to inspect
            image:
            class_index (int): Index of targeted class
            patch_size (int): Size of patch to apply on the image

        Returns:
            np.ndarray: Sensitivity map with shape (H, W, 3)
        """
        sensitivity_map = np.zeros(
            (
                math.ceil(image.shape[0] / patch_size),
                math.ceil(image.shape[1] / patch_size),
            )
        )

        patches = []
        coordinates = []
        for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size)):
            for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size)):
                patches.append(
                    apply_grey_patch(image, top_left_x, top_left_y, patch_size)
                )
                coordinates.append((index_y, index_x))

        predictions = model.predict(np.array(patches), batch_size=self.batch_size)
        target_class_predictions = [
            prediction[class_index] for prediction in predictions
        ]

        for (index_y, index_x), confidence in zip(
            coordinates, target_class_predictions
        ):
            sensitivity_map[index_y, index_x] = 1 - confidence

        return cv2.resize(sensitivity_map, image.shape[0:2])

    def save(self, grid, output_dir, output_name):
        """
        Save the output to a specific dir.

        Args:
            grid (numpy.ndarray): Grid of all heatmaps
            output_dir (str): Output directory path
            output_name (str): Output name
        """
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

        cv2.imwrite(
            str(Path(output_dir) / output_name), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        )
