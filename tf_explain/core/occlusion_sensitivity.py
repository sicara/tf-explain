import os
from pathlib import Path

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
        sensitivity_map = np.zeros((image.shape[0], image.shape[1]))

        for top_left_x in range(0, image.shape[0], patch_size):
            for top_left_y in range(0, image.shape[1], patch_size):
                confidence = OcclusionSensitivity.get_confidence_for_random_patch(
                    model=model,
                    image=image,
                    class_index=class_index,
                    top_left_x=top_left_x,
                    top_left_y=top_left_y,
                    patch_size=patch_size,
                )

                sensitivity_map[
                    top_left_y : top_left_y + patch_size,
                    top_left_x : top_left_x + patch_size,
                ] = (1 - confidence)

        return sensitivity_map

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
        os.makedirs(output_dir, exist_ok=True)

        grid_as_image = Image.fromarray((np.clip(grid, 0, 1) * 255).astype("uint8"))
        grid_as_image.save(Path(output_dir) / output_name)
