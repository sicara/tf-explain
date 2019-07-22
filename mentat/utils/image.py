""" Module for image operations """
import numpy as np


def apply_grey_patch(image, top_left_x, top_left_y, patch_size):
    """
    Replace a part of the image with a grey patch.

    Args:
        image (numpy.ndarray): Input image
        top_left_x (int): Top Left X position of the applied box
        top_left_y (int): Top Left Y position of the applied box
        patch_size (int): Size of patch to apply

    Returns:
        numpy.ndarray: Patched image
    """
    patched_image = np.array(image, copy=True)
    patched_image[
        top_left_y : top_left_y + patch_size, top_left_x : top_left_x + patch_size, :
    ] = 127.5

    return patched_image
