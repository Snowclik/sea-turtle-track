"""
src/synthetic_generation/transforms/geometric/random_perspective.py
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

from transforms.base import Transform, Polygon


class RandomPerspective(Transform):
    """
    Aplica una transformación de perspectiva aleatoria (homografía) 
    para simular el cabeceo (pitch/roll) del UAV.
    """

    def __init__(
        self,
        probability: float = 1.0,
        distortion_scale: float = 0.2,
        **kwargs,
    ):
        super().__init__(probability)
        self.distortion_scale = distortion_scale

    @property
    def category(self) -> str:
        return "geometric"

    def apply(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        h, w = image.shape[:2]
        
        dx = int(self.distortion_scale * w)
        dy = int(self.distortion_scale * h)
        
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([
            [rng.integers(0, dx + 1), rng.integers(0, dy + 1)],
            [rng.integers(w - dx, w + 1), rng.integers(0, dy + 1)],
            [rng.integers(0, dx + 1), rng.integers(h - dy, h + 1)],
            [rng.integers(w - dx, w + 1), rng.integers(h - dy, h + 1)]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        img_out = cv2.warpPerspective(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        polys_out = []
        for poly in polygons:
            new_poly = []
            for x_n, y_n in poly:
                px = x_n * w
                py = y_n * h
                
                denom = M[2, 0] * px + M[2, 1] * py + M[2, 2]
                if denom == 0:
                    denom = 1e-6
                    
                qx = (M[0, 0] * px + M[0, 1] * py + M[0, 2]) / denom
                qy = (M[1, 0] * px + M[1, 1] * py + M[1, 2]) / denom
                
                new_poly.append((min(1.0, max(0.0, qx / w)), min(1.0, max(0.0, qy / h))))
            polys_out.append(new_poly)
            
        return img_out, polys_out, {"distortion": round(self.distortion_scale, 4)}
