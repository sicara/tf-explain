import pytest
from tf_explain.core.vanilla_gradients import (
    VanillaGradients,
    UNSUPPORTED_ARCHITECTURE_WARNING,
)


def test_should_explain_output(
    convolutional_model_with_separate_activation_layer, random_data, mocker
):
    mocker.patch(
        "tf_explain.core.vanilla_gradients.grid_display", side_effect=lambda x: x
    )
    images, labels = random_data
    explainer = VanillaGradients()
    grid = explainer.explain(
        (images, labels), convolutional_model_with_separate_activation_layer, 0
    )

    # Outputs is in grayscale format
    assert grid.shape == images.shape[:-1]


def test_get_averaged_gradients(
    random_data, convolutional_model_without_final_activation
):
    images, _ = random_data
    gradients = VanillaGradients.compute_gradients(
        images, convolutional_model_without_final_activation, 0
    )

    assert gradients.shape == images.shape


def test_get_score_model_returns_suitable_model(
    convolutional_model_with_separate_activation_layer,
):
    explainer = VanillaGradients()
    # The last two layers of the original model are a Dense layer with no activation (i.e. linear)
    # followed by a Softmax layer.
    score_layer = convolutional_model_with_separate_activation_layer.layers[-2]
    softmax_layer = convolutional_model_with_separate_activation_layer.layers[-1]
    score_model = explainer.get_score_model(
        convolutional_model_with_separate_activation_layer
    )
    # The score model should exclude the final activation layer.
    assert score_model.layers[-1] == score_layer
    assert softmax_layer not in score_model.layers


def test_get_score_model_logs_warnings_when_model_not_suitable(convolutional_model):
    explainer = VanillaGradients()
    with pytest.warns(UserWarning, match=UNSUPPORTED_ARCHITECTURE_WARNING):
        explainer.get_score_model(convolutional_model)
