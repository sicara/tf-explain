""" Module for image operations """
import numpy as np
import tensorflow as tf


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


@tf.function
def transform_to_normalized_grayscale(tensor):
    """
    Transform tensor over RGB axis to grayscale.

    Args:
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, 3)

    Returns:
        tf.Tensor: 4D-Tensor of grayscale tensor, with shape (batch_size, H, W, 1)
    """
    grayscale_tensor = tf.reduce_sum(tensor, axis=-1)

    normalized_tensor = tf.cast(
        255 * tf.image.per_image_standardization(grayscale_tensor), tf.uint8
    )

    return normalized_tensor

def normalize_min_max(tensor):
    """
    Normalize tensor over RGB axis by subtracting min from maximum absolute values and dividing them by range.

    Args:
        tf.tensor: 4D-Tensor with shape (batch_size, H, W, 3)

    Returns:
        tf.Tensor: 2D-Tensor with shape (H, W)
    """

    normalized_tensor = tf.math.abs(tensor)
    normalized_tensor = tf.math.reduce_max(normalized_tensor, axis=-1)  # max along channels

    # Normalize to range between 0 and 1
    arr_min, arr_max = tf.math.reduce_min(normalized_tensor, axis=None), tf.math.reduce_max(normalized_tensor, axis=None)
    normalized_tensor = (normalized_tensor - arr_min) / (arr_max - arr_min + tf.constant(1e-16))
    normalized_tensor = tf.cast(255 * normalized_tensor, tf.uint8)[0]

    return normalized_tensor
