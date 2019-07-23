import os

import numpy as np

from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity


def test_should_get_sensitivity_map(random_data, mocker):
    mocker.patch(
        "tf_explain.core.occlusion_sensitivity.OcclusionSensitivity.get_confidence_for_random_patch",
        return_value=0.6,
    )

    x, y = random_data

    callback = OcclusionSensitivity()

    output = callback.get_sensitivity_map(
        model=None, image=x[0], class_index=None, patch_size=4
    )

    expected_output = 0.4 * np.ones((x[0].shape[0], x[0].shape[1]))

    np.testing.assert_almost_equal(output, expected_output)
