from typing import Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


import numpy as np
import numpy.random


@dataclass
class Camera:
    name: str = "gaia-camera-simulated"
    width: int = 1024  # pixels
    height: int = 1024  # pixels
    bin_x: int = 1  # binning factor in x
    bin_y: int = 1  # binning factor in y
    pitch: float = 13.5  # pixel pitch, microns
    plate_scale: float | None = (
        None  # arcsec/pixel (calculated from pitch+telescope if None)
    )
    max_adu: int = 2**16 - 1  # maximum ADU value
    well_depth: int = 2**16 - 1  # electrons
    bias: int = 300  # ADU
    gain: float = 1.0  # e-/ADU
    read_noise: float = 6.2  # e-
    dark_current: float = 0.2  # e-/s
    average_quantum_efficiency: float = 0.8  # fraction
    pixel_defects: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.pixel_defects:
            self.pixel_defects = {
                key: (
                    self._create_pixel_defect(key, **value)
                    if isinstance(value, dict)
                    else value
                )
                for key, value in self.pixel_defects.items()
            }

    def _create_pixel_defect(
        self,
        name: str,
        type: Literal["constant", "telegraphic", "noise"] = "constant",
        **kwargs,
    ) -> "PixelDefect":
        defect_classes = {
            "constant": ConstantPixelDefect,
            "telegraphic": TelegraphicPixelDefect,
            "noise": RandomNoisePixelDefect,
        }
        defect_type = type

        if defect_type not in defect_classes:
            raise ValueError(f"Unknown pixel defect type for {name}")

        return defect_classes[defect_type](name=name, **kwargs)

    @classmethod
    def create_ideal_camera(cls, **kwargs) -> "Camera":
        parameters = {
            "read_noise": 0,
            "dark_current": 0,
            "average_quantum_efficiency": 1.0,
            "bias": 0,
            "gain": 1.0,
        }
        return cls(**(parameters | kwargs))

    def _create_blank_image(self) -> np.ndarray:
        """Create a blank image filled with zeros for testing."""
        return np.zeros((self.height, self.width), dtype=np.uint16)


@dataclass
class PixelDefect(ABC):
    name: str = "defect"
    rate: float = 0.0
    seed: int = 0
    _rng: numpy.random.Generator | None = None
    _pixels: np.ndarray | None = None

    @property
    def pixels(self) -> np.ndarray:
        if self._pixels is None:
            raise ValueError(f"{self.name} pixels are not defined.")
        return self._pixels

    @property
    def rng(self) -> numpy.random.Generator:
        if self._rng is None:
            self._rng = numpy.random.default_rng(self.seed)

        return self._rng

    @abstractmethod
    def introduce_pixel_defect(self, image: np.ndarray, camera: Camera):
        """Introduce the defect into the image.

        Parameters
        ----------
        image : np.ndarray
            The image to which the defect will be introduced.
        camera : Camera
            The camera to which the defect applies.
        """
        raise NotImplementedError

    def set_pixels(self, pixels: np.ndarray, camera: Camera):
        """Set the pixels for the defect.

        Parameters
        ----------
        pixels : np.ndarray
            The pixel coordinates of the defect.
        camera : Camera
            The camera to which the defect applies.
        """
        self._check_pixel_bounds(pixels, camera.height, camera.width, self.name)
        self._pixels = pixels

    def number_of_defect_pixels(self, camera: Camera) -> int:
        return int(round(self.rate * camera.width * camera.height))

    def _select_random_pixels(self, camera: Camera) -> np.ndarray:
        return self.rng.integers(
            [camera.height, camera.width],
            size=(self.number_of_defect_pixels(camera), 2),
        )

    @staticmethod
    def _overwrite_pixel_values(
        image: np.ndarray, pixels: np.ndarray, pixel_values: int | np.ndarray
    ):
        if isinstance(pixel_values, np.ndarray):
            if not pixel_values.size == pixels.shape[0]:
                raise ValueError("Pixel values must match the number of pixels.")

        image[pixels[:, 0], pixels[:, 1]] = pixel_values

    @staticmethod
    def _check_pixel_bounds(pixels, height: int, width: int, name: str):
        if pixels is None:
            raise ValueError(f"{name} pixels are not defined.")
        if isinstance(pixels, list):
            pixels = np.array(pixels)
        if not pixels.ndim == 2 or not pixels.shape[1] == 2:
            raise ValueError(
                f"{name} pixels must be a numpy array of (x, y) tuples."
                f"Got shape {pixels.shape} instead."
            )
        if np.any(pixels[:, 0] >= height) or np.any(pixels[:, 1] >= width):
            raise ValueError(f"{name} pixels are outside the frame.")


@dataclass
class ConstantPixelDefect(PixelDefect):
    value: int = 0
    seed: int = 0

    def introduce_pixel_defect(self, image, camera):
        if self._pixels is None:
            self.set_pixels(self._select_random_pixels(camera), camera)

        self._overwrite_pixel_values(image, self.pixels, self.value)

        return image


@dataclass
class TelegraphicPixelDefect(PixelDefect):
    value: int = 0
    seed: int = 0
    dim: int = 0
    _lines: np.ndarray | None = None

    def introduce_pixel_defect(self, image, camera):
        if self._lines is None:
            self.set_lines(self._select_random_lines(camera, self.dim), camera)

        self._overwrite_pixel_values(image, self.pixels, self.value)

        return image

    @property
    def lines(self) -> np.ndarray:
        if self._lines is None:
            raise ValueError(f"{self.name} lines are not defined.")
        return self._lines

    def set_lines(self, lines: np.ndarray | list, camera: Camera):
        self._lines = np.array(lines)

        # Set pixels based on the selected lines
        if self.dim == 0:
            if not np.all(self._lines < camera.height):
                raise ValueError("Selected lines are outside the frame.")
            X, Y = np.meshgrid(np.arange(camera.width), self._lines)
        else:
            if not np.all(self._lines < camera.width):
                raise ValueError("Selected lines are outside the frame.")
            X, Y = np.meshgrid(self._lines, np.arange(camera.width))

        self.set_pixels(np.column_stack((X.ravel(), Y.ravel())), camera)

    def _select_random_lines(self, camera: Camera, dim: int = 0) -> np.ndarray:
        line_length = camera.width if dim == 0 else camera.height
        number_of_lines = self.number_of_defect_pixels(camera) // line_length
        selected_lines = self.rng.integers(line_length, size=(number_of_lines))
        return selected_lines


@dataclass
class RandomNoisePixelDefect(PixelDefect):
    noise_level: float = 10.0  # Standard deviation for Gaussian noise
    distribution: Literal["normal", "poisson"] = "normal"

    def introduce_pixel_defect(self, image, camera, seed: int | None = None):
        if seed is not None:
            self._rng = numpy.random.default_rng(seed)

        if self._pixels is None:
            self.set_pixels(self._select_random_pixels(camera), camera)

        if self.distribution == "poisson":
            noise = self.noise_level * self.rng.poisson(size=self.pixels.shape[0])
        elif self.distribution == "normal":
            noise = self.noise_level * self.rng.normal(size=self.pixels.shape[0])
        else:
            raise ValueError("Unknown noise distribution.")

        image[self.pixels[:, 0], self.pixels[:, 1]] += noise
        image = np.clip(image, 0, camera.max_adu)

        return image
