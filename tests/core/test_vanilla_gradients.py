import pytest
from tensorflow.keras.layers import Dense, Softmax
from tf_explain.core.vanilla_gradients import VanillaGradients


def test_should_explain_output(
    convolutional_model_for_vanilla_gradients, random_data, mocker
):
    mocker.patch(
        "tf_explain.core.vanilla_gradients.grid_display", side_effect=lambda x: x
    )
    images, labels = random_data
    explainer = VanillaGradients()
    grid = explainer.explain(
        (images, labels), convolutional_model_for_vanilla_gradients, 0
    )

    # Outputs is in grayscale format
    assert grid.shape == images.shape[:-1]


def test_get_averaged_gradients(random_data, score_model_for_vanilla_gradients):
    images, _ = random_data
    gradients = VanillaGradients.compute_gradients(
        images, score_model_for_vanilla_gradients, 0
    )

    assert gradients.shape == images.shape


def test_get_score_model_returns_suitable_model(
    convolutional_model_for_vanilla_gradients,
):
    explainer = VanillaGradients()
    # The last two layers of the original model are a Dense layer with no activation (i.e. linear)
    # followed by a Softmax layer.
    score_layer = convolutional_model_for_vanilla_gradients.layers[-2]
    softmax_layer = convolutional_model_for_vanilla_gradients.layers[-1]
    score_model = explainer._get_score_model(convolutional_model_for_vanilla_gradients)
    # The score model should exclude the final activation layer.
    assert score_model.layers[-1] == score_layer
    assert softmax_layer not in score_model.layers


def test_get_score_model_logs_warnings_when_model_not_suitable(convolutional_model):
    explainer = VanillaGradients()
    with pytest.warns(UserWarning, match=r"^Unsupported model architecture"):
        explainer._get_score_model(convolutional_model)
