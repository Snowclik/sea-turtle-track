"""
src/synthetic_generation/transforms/geometric/random_scale.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from transforms.base import Transform, Polygon


class RandomScale(Transform):
    """
    Simula cambios en la altitud del UAV escalando la imagen (zoom in / zoom out).
    La imagen se recorta al tamaño original (zoom in) o se le añade padding (zoom out).
    """

    def __init__(
        self,
        probability: float = 1.0,
        min_scale:   float = 0.8,
        max_scale:   float = 1.2,
        **kwargs,
    ):
        super().__init__(probability)
        self.min_scale = min_scale
        self.max_scale = max_scale

    @property
    def category(self) -> str:
        return "geometric"

    def apply(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        scale = float(rng.uniform(self.min_scale, self.max_scale))
        h, w = image.shape[:2]
        
        # Centro de la imagen
        cx, cy = w / 2.0, h / 2.0
        
        # Matriz afin
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        
        # Aplicar
        img_out = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Escalar polígonos
        def _scale_pt(x_n: float, y_n: float) -> Tuple[float, float]:
            px = x_n * w
            py = y_n * h
            qx = M[0, 0] * px + M[0, 1] * py + M[0, 2]
            qy = M[1, 0] * px + M[1, 1] * py + M[1, 2]
            return (min(1.0, max(0.0, qx / w)), min(1.0, max(0.0, qy / h)))

        polys_out = [[_scale_pt(x, y) for x, y in poly] for poly in polygons]
        
        return img_out, polys_out, {"scale": round(scale, 4)}
