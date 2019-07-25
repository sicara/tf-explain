import numpy as np

from tf_explain.core.grad_cam import GradCAM


def test_should_generate_ponderated_output(mocker):
    mocker.patch(
        "tf_explain.core.grad_cam.GradCAM.ponderate_output",
        side_effect=[mocker.sentinel.ponderated_1, mocker.sentinel.ponderated_2],
    )

    expected_output = [mocker.sentinel.ponderated_1, mocker.sentinel.ponderated_2]

    outputs = [mocker.sentinel.output_1, mocker.sentinel.output_2]
    grads = [mocker.sentinel.grads_1, mocker.sentinel.grads_2]

    output = GradCAM.generate_ponderated_output(outputs, grads)

    for real, expected in zip(output, expected_output):
        assert real == expected


def test_should_ponderate_output():
    grad = np.concatenate(
        [np.ones((3, 3, 1)), 2 * np.ones((3, 3, 1)), 3 * np.ones((3, 3, 1))], axis=-1
    )

    output = np.concatenate(
        [np.ones((3, 3, 1)), 2 * np.ones((3, 3, 1)), 4 * np.ones((3, 3, 1))], axis=-1
    )

    ponderated_output = GradCAM.ponderate_output(output, grad)

    ponderated_sum = 1 * 1 + 2 * 2 + 3 * 4
    expected_output = ponderated_sum * np.ones((3, 3))

    np.testing.assert_almost_equal(expected_output, ponderated_output)


def test_should_produce_gradients_and_filters(convolutional_model, random_data):
    images, _ = random_data
    layer_name = "activation_1"
    output, grads = GradCAM.get_gradients_and_filters(
        convolutional_model, images, layer_name, 0
    )

    assert output.shape == [len(images)] + list(
        convolutional_model.get_layer(layer_name).output.shape[1:]
    )
    assert grads.shape == output.shape


def test_should_explain_output(mocker):
    mock_get_gradients = mocker.patch(
        "tf_explain.core.grad_cam.GradCAM.get_gradients_and_filters"
    )
    mock_get_gradients.return_value = (
        [mocker.sentinel.conv_output_1, mocker.sentinel.conv_output_2],
        [mocker.sentinel.guided_grads_1, mocker.sentinel.guided_grads_2],
    )
    mock_generate_output = mocker.patch(
        "tf_explain.core.grad_cam.GradCAM.generate_ponderated_output"
    )
    mocker.sentinel.cam_1.numpy = lambda: mocker.sentinel.cam_1
    mocker.sentinel.cam_2.numpy = lambda: mocker.sentinel.cam_2
    mock_generate_output.return_value = [mocker.sentinel.cam_1, mocker.sentinel.cam_2]
    mocker.patch(
        "tf_explain.core.grad_cam.heatmap_display",
        side_effect=[mocker.sentinel.heatmap_1, mocker.sentinel.heatmap_2],
    )
    mocker.patch("tf_explain.core.grad_cam.grid_display", side_effect=lambda x: x)

    explainer = GradCAM()
    data = ([mocker.sentinel.image_1, mocker.sentinel.image_2], mocker.sentinel.labels)
    grid = explainer.explain(
        data,
        mocker.sentinel.model,
        mocker.sentinel.layer_name,
        mocker.sentinel.class_index,
    )

    for heatmap, expected_heatmap in zip(
        grid, [mocker.sentinel.heatmap_1, mocker.sentinel.heatmap_2]
    ):
        assert heatmap == expected_heatmap

    mock_get_gradients.assert_called_once_with(
        mocker.sentinel.model,
        [mocker.sentinel.image_1, mocker.sentinel.image_2],
        mocker.sentinel.layer_name,
        mocker.sentinel.class_index,
    )
    mock_generate_output.assert_called_once_with(
        [mocker.sentinel.conv_output_1, mocker.sentinel.conv_output_2],
        [mocker.sentinel.guided_grads_1, mocker.sentinel.guided_grads_2],
    )
