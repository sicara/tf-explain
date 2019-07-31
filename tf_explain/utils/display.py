""" Module for display related operations. """
import math
import warnings

import cv2
import numpy as np


def grid_display(array, num_rows=None, num_columns=None):
    """
    Display a list of images as a grid.

    Args:
        array (numpy.ndarray): 4D Tensor (batch_size, height, width, channels)

    Returns:
        numpy.ndarray: 3D Tensor as concatenation of input images on a grid
    """
    if num_rows is not None and num_columns is not None:
        total_grid_size = num_rows * num_columns
        if total_grid_size < len(array):
            warnings.warn(Warning("Given values for num_rows and num_columns doesn't allow to display all images. Values have been overrided to respect at least num_columns"))
            num_rows = math.ceil(len(array) / num_columns)
    elif num_rows is not None:
        num_columns = math.ceil(len(array) / num_rows)
    elif num_columns is not None:
        num_rows = math.ceil(len(array) / num_columns)
    else:
        num_rows = math.ceil(math.sqrt(len(array)))
        num_columns = math.ceil(math.sqrt(len(array)))

    number_of_missing_elements = num_columns * num_rows - len(array)
    # We fill the array with np.zeros elements to obtain a perfect square
    array = np.append(
        array,
        np.zeros((number_of_missing_elements, *array[0].shape)).astype(array.dtype),
        axis=0,
    )

    grid = np.concatenate(
        [
            np.concatenate(array[index * num_columns : (index + 1) * num_columns], axis=1)
            for index in range(num_rows)
        ],
        axis=0,
    )

    return grid


def filter_display(array, num_rows=None, num_columns=None):
    """
    Display a list of filter outputs as a greyscale images grid.

    Args:
        array (numpy.ndarray): 4D Tensor (batch_size, height, width, channels)

    Returns:
        numpy.ndarray: 3D Tensor as concatenation of input images on a grid
    """
    return grid_display(
        np.concatenate(np.rollaxis(array, 3, 1), axis=0), num_rows, num_columns
    )


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
