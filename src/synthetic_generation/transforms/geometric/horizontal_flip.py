"""
src/synthetic_generation/transforms/geometric/horizontal_flip.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from transforms.base import Transform, Polygon


class HorizontalFlip(Transform):
    """
    Voltea horizontalmente la imagen y refleja los polígonos:
        x_new = 1 - x,  y_new = y
    """

    @property
    def category(self) -> str:
        return "geometric"

    def apply(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        img_out   = cv2.flip(image, 1)
        polys_out = [[(1.0 - x, y) for x, y in poly] for poly in polygons]
        return img_out, polys_out, {}
