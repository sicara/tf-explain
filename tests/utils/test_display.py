import numpy as np
import pytest

from tf_explain.utils.display import grid_display, heatmap_display


def test_should_fill_with_zeros_if_no_perfect_square():
    array = np.random.random((7, 28, 28, 3))

    grid = grid_display(array)

    assert grid.shape == (28 * 3, 28 * 3, 3)
    np.testing.assert_equal(grid[56:, 28:, :], np.zeros((28, 56, 3)))


def test_should_reshape_input_array_as_a_grid():
    array = np.array(
        [np.ones((4, 4)), np.ones((4, 4)), np.zeros((4, 4)), np.zeros((4, 4))]
    )

    grid = grid_display(array)

    expected_grid = np.concatenate([np.ones((4, 8)), np.zeros((4, 8))], axis=0)

    np.testing.assert_array_equal(grid, expected_grid)


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
