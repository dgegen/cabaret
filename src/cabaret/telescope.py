from dataclasses import dataclass

import numpy as np


@dataclass
class Telescope:
    focal_length: float = 8.0  # meters
    diameter: float = 1.0  # meters
    collecting_area: float | None = None  # square meters

    def __post_init__(self):
        if self.collecting_area is None:
            self.collecting_area = np.pi * (self.diameter / 2) ** 2  # square meters
