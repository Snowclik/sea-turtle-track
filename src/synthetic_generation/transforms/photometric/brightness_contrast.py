"""
src/synthetic_generation/transforms/photometric/brightness_contrast.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any

import numpy as np

from transforms.base import Transform, Polygon


class BrightnessContrast(Transform):
    """
    Ajusta el brillo y el contraste de la imagen.
    Transformación fotométrica pura: no modifica los polígonos.

    Fórmula: pixel_out = clip(pixel * contrast + brightness, 0, 255)
    """

    def __init__(
        self,
        probability:      float = 1.0,
        brightness_delta: int   = 30,
        contrast_min:     float = 0.8,
        contrast_max:     float = 1.2,
        **kwargs,
    ):
        super().__init__(probability)
        self.brightness_delta = brightness_delta
        self.contrast_min     = contrast_min
        self.contrast_max     = contrast_max

    @property
    def category(self) -> str:
        return "photometric"

    def apply(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        brightness = int(rng.integers(-self.brightness_delta, self.brightness_delta + 1))
        contrast   = float(rng.uniform(self.contrast_min, self.contrast_max))

        img_out = np.clip(
            image.astype(np.float32) * contrast + brightness,
            0, 255,
        ).astype(np.uint8)

        return img_out, polygons, {
            "brightness": brightness,
            "contrast":   round(contrast, 4),
        }
