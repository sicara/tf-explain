""" Module for display related operations. """
import math
import warnings

import numpy as np


def grid_display(array):
    """
    Display a list of images as a grid.

    Args:
        array (numpy.ndarray): 4D Tensor (batch_size, height, width, channels)

    Returns:
        numpy.ndarray: 3D Tensor as concatenation of input images on a grid
    """
    grid_size = int(math.sqrt(len(array)))

    if grid_size != math.sqrt(len(array)):
        warnings.warn('Elements to display are not a perfect square. '
                      'Last elements will be truncated.')

    grid = np.concatenate([
        np.concatenate(array[index*grid_size: (index+1)*grid_size], axis=1)
        for index in range(grid_size)
    ], axis=0)

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
