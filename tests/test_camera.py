import numpy as np
import pytest

from cabaret.camera import Camera, ConstantPixelDefect


def test_camera_with_pixel_defects():
    defects = {"hot": {"rate": 0.1, "seed": 42}, "cold": {"rate": 0.05, "seed": 43}}
    camera = Camera(pixel_defects=defects)
    assert len(camera.pixel_defects) == 2
    assert all(
        isinstance(defect, ConstantPixelDefect)
        for defect in camera.pixel_defects.values()
    )


def test_constant_pixel_defect_set_pixels():
    camera = Camera(width=10, height=10)
    defect = ConstantPixelDefect(value=255, seed=42)

    # Valid case
    valid_pixels = np.array([[0, 0], [1, 1], [2, 2]])
    defect.set_pixels(valid_pixels, camera)
    assert np.array_equal(defect.pixels, valid_pixels)

    # Invalid case: Out of bounds
    with pytest.raises(ValueError, match="defect pixels are outside the frame."):
        out_of_bounds_pixels = np.array([[10, 10]])
        defect.set_pixels(out_of_bounds_pixels, camera)


if __name__ == "__main__":
    pytest.main()
