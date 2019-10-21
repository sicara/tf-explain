from tf_explain.core.vanilla_gradients import VanillaGradients


def test_should_explain_output(convolutional_model, random_data, mocker):
    mocker.patch(
        "tf_explain.core.vanilla_gradients.grid_display", side_effect=lambda x: x
    )
    images, labels = random_data
    explainer = VanillaGradients()
    grid = explainer.explain((images, labels), convolutional_model, 0)

    # Outputs is in grayscale format
    assert grid.shape == images.shape[:-1]


def test_get_averaged_gradients(random_data, convolutional_model):
    images, _ = random_data
    gradients = VanillaGradients.compute_gradients(images, convolutional_model, 0)

    assert gradients.shape == images.shape
