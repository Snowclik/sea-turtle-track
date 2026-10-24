"""
src/synthetic_generation/transforms/photometric/motion_blur.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from transforms.base import Transform, Polygon


class MotionBlur(Transform):
    """
    Simula la vibración del dron o desenfoque por movimiento rápido.
    Aplica un filtro lineal direccional.
    Transformación fotométrica: no altera las coordenadas de los polígonos.
    """

    def __init__(
        self,
        probability: float = 1.0,
        kernel_size: int   = 15,
        **kwargs,
    ):
        super().__init__(probability)
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    @property
    def category(self) -> str:
        return "photometric"

    def apply(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        angle = float(rng.uniform(0, 180))
        
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        
        x1 = int(center - center * np.cos(np.deg2rad(angle)))
        y1 = int(center - center * np.sin(np.deg2rad(angle)))
        x2 = int(center + center * np.cos(np.deg2rad(angle)))
        y2 = int(center + center * np.sin(np.deg2rad(angle)))
        
        cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
        
        if np.sum(kernel) == 0:
            kernel[center, center] = 1
        else:
            kernel = kernel / np.sum(kernel)
        
        img_out = cv2.filter2D(image, -1, kernel)
        
        return img_out, polygons, {"kernel_size": self.kernel_size, "angle": round(angle, 4)}
