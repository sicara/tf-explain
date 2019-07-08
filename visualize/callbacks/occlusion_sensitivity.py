import os
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import Callback

from visualize.utils.display import grid_display
from visualize.utils.image import apply_grey_patch


class OcclusionSensitivityCallback(Callback):
    def __init__(self, validation_data, patch_size, class_index, output_dir=Path('./logs/occlusion_sensitivity')):
        super(OcclusionSensitivityCallback, self).__init__()
        self.validation_data = validation_data
        self.patch_size = patch_size
        self.class_index = class_index
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        sensitivity_maps = np.array([
            self.get_sensitivity_map(self.model, image, self.class_index, self.patch_size)
            for image in self.validation_data[0]
        ])

        grid = grid_display(sensitivity_maps)

        im = Image.fromarray((np.clip(grid, 0, 1) * 255).astype('uint8'))
        im.save(Path(self.output_dir) / f'{epoch}.png')

    @staticmethod
    def get_sensitivity_map(model, image, class_index, patch_size):
        sensitivity_map = np.zeros((image.shape[0], image.shape[1]))

        for top_left_x in range(0, image.shape[0], patch_size):
            for top_left_y in range(0, image.shape[1], patch_size):
                confidence = OcclusionSensitivityCallback.get_confidence_for_random_patch(
                    model=model,
                    image=image,
                    class_index=class_index,
                    top_left_x=top_left_x,
                    top_left_y=top_left_y,
                    patch_size=patch_size,
                )
                print(confidence, flush=True)
                sensitivity_map[
                    top_left_y:top_left_y + patch_size,
                    top_left_x:top_left_x + patch_size,
                ] = 1 - confidence

        return sensitivity_map

    @staticmethod
    def get_confidence_for_random_patch(model, image, class_index, top_left_x, top_left_y, patch_size):
        """
        Get class confidence for input image with a patch applied.

        Args:
            model:
            image:
            class_index:
            top_left_x:
            top_left_y:
            patch_size:

        Returns:

        """
        patch = apply_grey_patch(image, top_left_x, top_left_y, patch_size)
        predicted_classes = model.predict(np.array([patch]))[0]
        confidence = predicted_classes[class_index]

        return confidence

