import numpy as np

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


def test_get_averaged_gradients(random_data, convolutional_model):
    images, _ = random_data
    num_samples = 2
    gradients = SmoothGrad.get_averaged_gradients(
        images, convolutional_model, 0, num_samples=num_samples
    )

    expected_output_shape = (images.shape[0] / num_samples, *images.shape[1:])

    assert gradients.shape == expected_output_shape
