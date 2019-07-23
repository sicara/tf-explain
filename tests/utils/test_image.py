import numpy as np

from tf_explain.utils.image import apply_grey_patch


def test_should_apply_grey_patch_on_image():
    input_image = np.zeros((10, 20, 3))

    output = apply_grey_patch(input_image, 0, 0, 10)

    expected_output = np.concatenate(
        [127.5 * np.ones((10, 10, 3)), np.zeros((10, 10, 3))], axis=1
    )

    np.testing.assert_almost_equal(output, expected_output)
