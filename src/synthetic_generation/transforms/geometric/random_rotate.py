"""
src/synthetic_generation/transforms/geometric/random_rotate.py
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from transforms.base import Transform, Polygon


class RandomRotate(Transform):
    """
    Rota la imagen un ángulo aleatorio dentro de [min_angle, max_angle].
    El polígono se rota alrededor del centro de la imagen.
    Las coordenadas resultantes se recortan a [0, 1].
    """

    def __init__(
        self,
        probability: float = 1.0,
        min_angle:   float = -15,
        max_angle:   float =  15,
        **kwargs,
    ):
        super().__init__(probability)
        self.min_angle = min_angle
        self.max_angle = max_angle

    @property
    def category(self) -> str:
        return "geometric"

    def apply(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        angle    = float(rng.uniform(self.min_angle, self.max_angle))
        h, w     = image.shape[:2]
        cx, cy   = w / 2.0, h / 2.0

        M        = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img_out  = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Rotar polígonos alrededor del centro
        rad   = math.radians(-angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        def _rotate(x_n: float, y_n: float) -> Tuple[float, float]:
            px = x_n * w - cx
            py = y_n * h - cy
            qx = (cos_a * px - sin_a * py + cx) / w
            qy = (sin_a * px + cos_a * py + cy) / h
            return (min(1.0, max(0.0, qx)), min(1.0, max(0.0, qy)))

        polys_out = [[_rotate(x, y) for x, y in poly] for poly in polygons]
        return img_out, polys_out, {"angle": round(angle, 4)}
