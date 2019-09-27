from pathlib import Path

import numpy as np

from tf_explain.utils.saver import save_grayscale, save_rgb


def test_should_save_grayscale_image(output_dir):
    input_image = np.ones((28, 28, 1), dtype="uint8")

    save_grayscale(input_image, output_dir, "grayscale.png")

    assert len(list(Path(output_dir).iterdir())) == 1


def test_should_save_rgb_image(output_dir):
    input_image = np.ones((28, 28, 3), dtype="uint8")

    save_rgb(input_image, output_dir, "rgb.png")

    assert len(list(Path(output_dir).iterdir())) == 1
