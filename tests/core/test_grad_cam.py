import numpy as np

from tf_explain.core.grad_cam import GradCAM


def test_should_generate_ponderated_output():
    grads = np.concatenate(
        [np.ones((3, 3, 1)), 2 * np.ones((3, 3, 1)), 3 * np.ones((3, 3, 1))], axis=-1
    )

    outputs = np.concatenate(
        [np.ones((3, 3, 1)), 2 * np.ones((3, 3, 1)), 4 * np.ones((3, 3, 1))], axis=-1
    )

    cam = GradCAM.generate_ponderated_output(outputs, grads)

    ponderated_sum = 1 * 1 + 2 * 2 + 3 * 4
    expected_output = (ponderated_sum + 1) * np.ones((3, 3))

    np.testing.assert_almost_equal(expected_output, cam)


def test_should_produce_gradients_and_filters(convolutional_model, random_data):
    images, _ = random_data
    layer_name = "activation_1"
    output, grads = GradCAM.get_gradients_and_filters(
        convolutional_model, images, layer_name, 0
    )

    assert output.shape == convolutional_model.get_layer(layer_name).output.shape[1:]
    assert grads.shape == output.shape


def test_should_explain_output(mocker):
    mock_get_gradients = mocker.patch(
        "tf_explain.core.grad_cam.GradCAM.get_gradients_and_filters"
    )
    mock_get_gradients.return_value = (
        mocker.sentinel.output,
        mocker.sentinel.guided_grads,
    )
    mock_generate_output = mocker.patch(
        "tf_explain.core.grad_cam.GradCAM.generate_ponderated_output"
    )
    mock_generate_output.return_value = mocker.sentinel.cam
    mocker.patch(
        "tf_explain.core.grad_cam.heatmap_display", return_value=mocker.sentinel.heatmap
    )

    explainer = GradCAM()
    data = ([mocker.sentinel.image], mocker.sentinel.labels)
    heatmap = explainer.explain(
        data,
        mocker.sentinel.model,
        mocker.sentinel.layer_name,
        mocker.sentinel.class_index,
    )

    assert heatmap == mocker.sentinel.heatmap

    mock_get_gradients.assert_called_once_with(
        mocker.sentinel.model,
        [mocker.sentinel.image],
        mocker.sentinel.layer_name,
        mocker.sentinel.class_index,
    )
    mock_generate_output.assert_called_once_with(
        mocker.sentinel.output, mocker.sentinel.guided_grads
    )
