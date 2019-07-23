import numpy as np

from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity


def test_should_get_sensitivity_map(random_data, mocker):
    mocker.patch(
        "tf_explain.core.occlusion_sensitivity.OcclusionSensitivity.get_confidence_for_random_patch",
        return_value=0.6,
    )
    mocker.patch(
        "tf_explain.core.occlusion_sensitivity.cv2.resize", side_effect=lambda x, _: x
    )

    x, y = random_data
    patch_size = 4

    output = OcclusionSensitivity.get_sensitivity_map(
        model=None, image=x[0], class_index=None, patch_size=patch_size
    )

    expected_output = 0.4 * np.ones(
        (x[0].shape[0] // patch_size, x[0].shape[1] // patch_size)
    )

    np.testing.assert_almost_equal(output, expected_output)


def test_should_get_confidence_for_patch(convolutional_model, mocker):
    convolutional_model.predict = mocker.MagicMock(return_value=[[1, 0]])
    mocker.patch(
        "tf_explain.core.occlusion_sensitivity.apply_grey_patch",
        return_value=np.random.randint(
            low=0, high=255, size=convolutional_model.inputs[0].shape[1:]
        ),
    )

    class_index = 0
    confidence = OcclusionSensitivity.get_confidence_for_random_patch(
        convolutional_model, None, class_index, None, None, None
    )

    assert confidence == 1


def test_should_produce_heatmap(convolutional_model, random_data, mocker):
    mocker.patch(
        "tf_explain.core.occlusion_sensitivity.grid_display",
        return_value=mocker.sentinel.grid,
    )

    explainer = OcclusionSensitivity()
    grid = explainer.explain(random_data, convolutional_model, 0, 10)

    assert grid == mocker.sentinel.grid
