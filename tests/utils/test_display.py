import numpy as np
import pytest

from tf_explain.utils.display import grid_display, heatmap_display, image_to_uint_255


@pytest.fixture(scope="session")
def array_to_display():
    return np.random.random((7, 28, 28, 3))


@pytest.mark.parametrize(
    "grid_display_kwargs,expected_shape",
    [
        ({}, (28 * 3, 28 * 3, 3)),
        ({"num_columns": 10, "num_rows": 10}, (28 * 10, 28 * 10, 3)),
        ({"num_columns": 7}, (28, 28 * 7, 3)),
        ({"num_rows": 7}, (28 * 7, 28, 3)),
    ],
)
def test_should_return_a_grid_square_by_default(
    grid_display_kwargs, expected_shape, array_to_display
):
    grid = grid_display(array_to_display, **grid_display_kwargs)

    assert grid.shape == expected_shape


def test_should_raise_warning_if_grid_size_is_too_small(array_to_display):
    with pytest.warns(Warning) as w:
        grid = grid_display(array_to_display, num_rows=1, num_columns=1)

    assert (
        w[0].message.args[0]
        == "Given values for num_rows and num_columns doesn't allow to display all images. Values have been overrided to respect at least num_columns"
    )
    assert grid.shape == (28 * 7, 28, 3)


def test_should_fill_with_zeros_if_missing_elements(array_to_display):
    grid = grid_display(array_to_display)

    assert grid.shape == (28 * 3, 28 * 3, 3)
    np.testing.assert_equal(grid[56:, 28:, :], np.zeros((28, 56, 3)))


@pytest.mark.parametrize(
    "input_image,expected_output",
    [
        (np.ones((28, 28)) / 255.0, np.ones((28, 28)).astype("uint8")),
        (-np.ones((28, 28)) / 255.0, 127 * np.ones((28, 28)).astype("uint8")),
        (
            10 * np.ones((28, 28)).astype("uint8"),
            10 * np.ones((28, 28)).astype("uint8"),
        ),
    ],
)
def test_should_transform_to_uint_255_image(input_image, expected_output):
    output = image_to_uint_255(input_image)

    np.testing.assert_almost_equal(output, expected_output)


def test_should_display_heatmap(mocker):
    mock_add_weighted = mocker.patch("tf_explain.utils.display.cv2.addWeighted")
    mock_apply_color_map = mocker.patch("tf_explain.utils.display.cv2.applyColorMap")
    mock_cvt_color = mocker.patch(
        "tf_explain.utils.display.cv2.cvtColor", return_value=mocker.sentinel.heatmap
    )

    heatmap = np.random.random((3, 3))
    original_image = np.zeros((10, 10, 3))

    output = heatmap_display(heatmap, original_image)

    assert output == mocker.sentinel.heatmap
    assert mock_add_weighted.call_count == 1
    assert mock_apply_color_map.call_count == 1
    assert mock_cvt_color.call_count == 3
