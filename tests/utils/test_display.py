import numpy as np
import pytest

from mentat.utils.display import grid_display, heatmap_display


def test_should_raise_warning_if_no_perfect_square():
    array = np.random.random((7, 28, 28, 3))

    with pytest.warns(Warning):
        grid = grid_display(array)

    assert grid.shape == (56, 56, 3)


def test_should_reshape_input_array_as_a_grid():
    array = np.array([
        np.ones((4, 4)),
        np.ones((4, 4)),
        np.zeros((4, 4)),
        np.zeros((4, 4)),
    ])

    grid = grid_display(array)

    expected_grid = np.concatenate([
        np.ones((4, 8)),
        np.zeros((4, 8))
    ], axis=0)

    np.testing.assert_array_equal(grid, expected_grid)


def test_should_display_heatmap(mocker):
    mock_addweighted = mocker.patch('mentat.utils.display.cv2.addWeighted')
    mock_addweighted.return_value=mocker.sentinel.heatmap

    heatmap = np.random.random((3, 3, 3))
    original_image = np.zeros((10, 10, 3))

    output = heatmap_display(heatmap, original_image)

    assert output == mocker.sentinel.heatmap
