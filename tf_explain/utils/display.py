""" Module for display related operations. """
import math

import cv2
import numpy as np


def grid_display(array):
    """
    Display a list of images as a grid.

    Args:
        array (numpy.ndarray): 4D Tensor (batch_size, height, width, channels)

    Returns:
        numpy.ndarray: 3D Tensor as concatenation of input images on a grid
    """
    grid_size = math.ceil(math.sqrt(len(array)))

    # We fill the array with np.zeros elements to obtain a perfect square
    number_of_missing_elements = grid_size ** 2 - len(array)
    array = np.append(
        array,
        np.zeros((number_of_missing_elements, *array[0].shape)).astype(array.dtype),
        axis=0,
    )

    grid = np.concatenate(
        [
            np.concatenate(array[index * grid_size : (index + 1) * grid_size], axis=1)
            for index in range(grid_size)
        ],
        axis=0,
    )

    return grid


def filter_display(array):
    """
    Display a list of filter outputs as a greyscale images grid.

    Args:
        array (numpy.ndarray): 4D Tensor (batch_size, height, width, channels)

    Returns:
        numpy.ndarray: 3D Tensor as concatenation of input images on a grid
    """
    return grid_display(np.concatenate(np.rollaxis(array, 3, 1), axis=0))


def heatmap_display(heatmap, original_image):
    """
    Apply a heatmap (as an np.ndarray) on top of an original image.

    Args:
        heatmap (numpy.ndarray): Array corresponding to the heatmap
        original_image (numpy.ndarray): Image on which we apply the heatmap

    Returns:
        np.ndarray: Original image with heatmap applied
    """
    map = cv2.resize(heatmap, original_image.shape[0:2])

    map = (map - np.min(map)) / (map.max() - map.min())

    heatmap = cv2.applyColorMap(
        cv2.cvtColor((map * 255).astype("uint8"), cv2.COLOR_GRAY2BGR), cv2.COLORMAP_JET
    )

    output = cv2.addWeighted(
        cv2.cvtColor(original_image.astype("uint8"), cv2.COLOR_RGB2BGR),
        0.7,
        heatmap,
        1,
        0,
    )

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
