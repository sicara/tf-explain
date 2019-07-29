import numpy as np
import tensorflow as tf

from tf_explain.core.smoothgrad import SmoothGrad


def test_should_explain_output(convolutional_model, random_data, mocker):
    mocker.patch("tf_explain.core.smoothgrad.grid_display", side_effect=lambda x: x)
    images, labels = random_data
    explainer = SmoothGrad()
    grid = explainer.explain((images, labels), convolutional_model, 0)

    # Outputs is in grayscale format
    assert grid.shape == images.shape[:-1]


def test_generate_noisy_images(mocker):
    input_shape = (10, 28, 28, 1)
    num_samples = 3
    mocker.patch(
        "tf_explain.core.smoothgrad.np.random.normal",
        return_value=np.ones((30, 28, 28, 1)),
    )

    images = np.ones(input_shape)
    output = SmoothGrad.generate_noisy_images(images, num_samples=num_samples, noise=1)

    np.testing.assert_array_equal(output, 2 * np.ones((30, 28, 28, 1)))


def test_should_transform_gradients_to_grayscale():
    gradients = tf.random.uniform((4, 28, 28, 3))

    grayscale_gradients = SmoothGrad.transform_to_grayscale(gradients)
    expected_output_shape = (4, 28, 28)

    assert grayscale_gradients.shape == expected_output_shape


def test_get_averaged_gradients(random_data, convolutional_model):
    images, _ = random_data
    num_samples = 2
    gradients = SmoothGrad.get_averaged_gradients(
        images, convolutional_model, 0, num_samples=num_samples
    )

    expected_output_shape = (images.shape[0] / num_samples, *images.shape[1:])

    assert gradients.shape == expected_output_shape
