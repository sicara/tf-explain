from pathlib import Path

import cv2


def save_grayscale(grid, output_dir, output_name):
    Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

    cv2.imwrite(str(Path(output_dir) / output_name), grid)


def save_rgb(grid, output_dir, output_name):
    Path.mkdir(Path(output_dir), parents=True, exist_ok=True)

    cv2.imwrite(
        str(Path(output_dir) / output_name), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    )
