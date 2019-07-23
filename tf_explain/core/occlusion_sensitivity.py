import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tf_explain.utils.display import grid_display
from tf_explain.utils.image import apply_grey_patch


class OcclusionSensitivity:

    """
    Perform Grad CAM algorithm for a given input

    Paper: [Grad-CAM: Visual Explanations from Deep Networks
            via Gradient-based Localization](https://arxiv.org/abs/1610.02391)
    """

    def __init__(self, batch_size=None):
        self.batch_size = batch_size

    def explain(self, validation_data, model, class_index, patch_size):
        images, _ = validation_data
        sensitivity_maps = np.array(
            [
                self.get_sensitivity_map(model, image, class_index, patch_size)
                for image in images
            ]
        )

        grid = grid_display(sensitivity_maps)

        return grid

    def get_sensitivity_map(self, model, image, class_index, patch_size):
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
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

        grid_as_image = Image.fromarray((np.clip(grid, 0, 1) * 255).astype("uint8"))
        grid_as_image.save(Path(output_dir) / output_name)
