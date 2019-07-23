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

    @staticmethod
    def get_sensitivity_map(model, image, class_index, patch_size):
        sensitivity_map = np.zeros(
            (
                math.ceil(image.shape[0] / patch_size),
                math.ceil(image.shape[1] / patch_size),
            )
        )

        for index_x, top_left_x in enumerate(range(0, image.shape[0], patch_size)):
            for index_y, top_left_y in enumerate(range(0, image.shape[1], patch_size)):
                confidence = OcclusionSensitivity.get_confidence_for_random_patch(
                    model=model,
                    image=image,
                    class_index=class_index,
                    top_left_x=top_left_x,
                    top_left_y=top_left_y,
                    patch_size=patch_size,
                )

                sensitivity_map[index_y, index_x] = 1 - confidence

        return cv2.resize(sensitivity_map, image.shape[0:2])

    @staticmethod
    def get_confidence_for_random_patch(
        model, image, class_index, top_left_x, top_left_y, patch_size
    ):
        """
        Get class confidence for input image with a patch applied.

        Args:
            model (tensorflow.keras.Model): Tensorflow Model
            image (numpy.ndarray): Image to predict
            class_index (int): Target class
            top_left_x (int): Coordinate x for grey patch
            top_left_y (int): Coordinate y for grey patch
            patch_size (int): Size of grey patch to apply on image

        Returns:
            float: Confidence for prediction of patched image.
        """
        patch = apply_grey_patch(image, top_left_x, top_left_y, patch_size)
        predicted_classes = model.predict(np.array([patch]))[0]
        confidence = predicted_classes[class_index]

        return confidence

    def save(self, grid, output_dir, output_name):
        Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

        grid_as_image = Image.fromarray((np.clip(grid, 0, 1) * 255).astype("uint8"))
        grid_as_image.save(Path(output_dir) / output_name)
