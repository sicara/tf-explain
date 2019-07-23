import math

import numpy as np

from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity


def test_should_get_sensitivity_map(convolutional_model, random_data, mocker):
    x, y = random_data
    patch_size = 4

    predict_return_value = np.ones(
        (
            math.ceil(x[0].shape[0] / patch_size)
            * math.ceil(x[0].shape[1] / patch_size),
            1,
        )
    ) * np.expand_dims([0.6, 0.4], axis=0)
    convolutional_model.predict = mocker.MagicMock(return_value=predict_return_value)
    mocker.patch(
        "tf_explain.core.occlusion_sensitivity.apply_grey_patch",
        return_value=np.random.randint(
            low=0, high=255, size=convolutional_model.inputs[0].shape[1:]
        ),
    )
    mocker.patch(
        "tf_explain.core.occlusion_sensitivity.cv2.resize", side_effect=lambda x, _: x
    )

    output = OcclusionSensitivity().get_sensitivity_map(
        model=convolutional_model, image=x[0], class_index=0, patch_size=patch_size
    )

    expected_output = 0.4 * np.ones(
        (x[0].shape[0] // patch_size, x[0].shape[1] // patch_size)
    )

    np.testing.assert_almost_equal(output, expected_output)


def test_should_produce_heatmap(convolutional_model, random_data, mocker):
    mocker.patch(
        "tf_explain.core.occlusion_sensitivity.grid_display",
        return_value=mocker.sentinel.grid,
    )

    explainer = OcclusionSensitivity()
    grid = explainer.explain(random_data, convolutional_model, 0, 10)

    assert grid == mocker.sentinel.grid
