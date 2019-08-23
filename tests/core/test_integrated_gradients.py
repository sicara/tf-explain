import numpy as np

from tf_explain.core.integrated_gradients import IntegratedGradients


def test_should_explain_output(convolutional_model, random_data, mocker):
    mocker.patch(
        "tf_explain.core.integrated_gradients.grid_display", side_effect=lambda x: x
    )
    images, labels = random_data
    explainer = IntegratedGradients()
    grid = explainer.explain((images, labels), convolutional_model, 0)

    # Outputs is in grayscale format
    assert grid.shape == images.shape[:-1]


def test_generate_linear_path():
    input_shape = (28, 28, 1)
    target = np.ones(input_shape)
    baseline = np.zeros(input_shape)
    n_steps = 3

    expected_output = [baseline, 1 / 2 * (target - baseline), target]

    output = IntegratedGradients.generate_linear_path(baseline, target, n_steps)

    np.testing.assert_almost_equal(output, expected_output)


def test_get_integrated_gradients(random_data, convolutional_model):
    images, _ = random_data
    n_steps = 4
    gradients = IntegratedGradients.get_integrated_gradients(
        images, convolutional_model, 0, n_steps=n_steps
    )

    expected_output_shape = (images.shape[0] / n_steps, *images.shape[1:])

    assert gradients.shape == expected_output_shape
