"""
src/synthetic_generation/transforms.py
─────────────────────────────────────
Única responsabilidad: definir la clase base Transform y las
transformaciones piloto de la Etapa 3.

Contrato de toda transformación:
  - Recibe: imagen (np.ndarray), polígonos (List[Polygon]), rng (np.random.Generator)
  - Devuelve: (imagen_transformada, polígonos_transformados, params_aplicados)
  - NUNCA convierte polígonos a bboxes.
  - Siempre devuelve los polígonos en coordenadas normalizadas [0, 1].

Transformaciones piloto:
  1. HorizontalFlip
  2. RandomRotate
  3. BrightnessContrast  ← solo píxeles, no afecta polígonos
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import math

import cv2
import numpy as np

# Tipo alias
Polygon = List[Tuple[float, float]]


# ── Clase Base ────────────────────────────────────────────────────────────────

class Transform(ABC):
    """
    Interfaz común para todas las transformaciones del motor ASDGE.
    Subclases deben implementar `apply()`.
    """

    def __init__(self, probability: float = 1.0, **kwargs):
        if not (0.0 <= probability <= 1.0):
            raise ValueError(f"probability debe estar en [0, 1], recibido: {probability}")
        self.probability = probability

    def __call__(
        self,
        image: np.ndarray,
        polygons: List[Polygon],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any], bool]:
        """
        Decide si aplicar la transformación según la probabilidad.
        Devuelve (img, polys, params, applied).
        """
        if rng.random() <= self.probability:
            img_out, polys_out, params = self.apply(image, polygons, rng)
            return img_out, polys_out, params, True
        return image, polygons, {}, False

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        polygons: List[Polygon],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        """Aplica la transformación. Siempre se ejecuta (prob ya evaluada)."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ── Transformación 1: HorizontalFlip ─────────────────────────────────────────

class HorizontalFlip(Transform):
    """
    Voltea horizontalmente la imagen y refleja los polígonos:
        x_new = 1 - x
        y_new = y
    """

    def apply(
        self,
        image: np.ndarray,
        polygons: List[Polygon],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        img_out   = cv2.flip(image, 1)
        polys_out = [[(1.0 - x, y) for x, y in poly] for poly in polygons]
        return img_out, polys_out, {}


# ── Transformación 2: RandomRotate ───────────────────────────────────────────

class RandomRotate(Transform):
    """
    Rota la imagen un ángulo aleatorio dentro de [min_angle, max_angle].
    El polígono se rota alrededor del centro de la imagen.
    Las coordenadas fuera de [0, 1] se recortan (clip).
    """

    def __init__(self, probability: float = 1.0, min_angle: float = -15, max_angle: float = 15, **kwargs):
        super().__init__(probability)
        self.min_angle = min_angle
        self.max_angle = max_angle

    def apply(
        self,
        image: np.ndarray,
        polygons: List[Polygon],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        angle = float(rng.uniform(self.min_angle, self.max_angle))
        h, w  = image.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Matriz de rotación (OpenCV)
        M     = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img_out = cv2.warpAffine(image, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT_101)

        # Rotar polígonos
        rad   = math.radians(-angle)    # OpenCV rota en sentido horario
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        def _rotate_point(x_norm: float, y_norm: float) -> Tuple[float, float]:
            # Desnormalizar
            px, py = x_norm * w, y_norm * h
            # Rotar alrededor del centro
            qx = cos_a * (px - cx) - sin_a * (py - cy) + cx
            qy = sin_a * (px - cx) + cos_a * (py - cy) + cy
            # Renormalizar y clip
            return (min(1.0, max(0.0, qx / w)),
                    min(1.0, max(0.0, qy / h)))

        polys_out = [[_rotate_point(x, y) for x, y in poly] for poly in polygons]
        return img_out, polys_out, {"angle": round(angle, 4)}


# ── Transformación 3: BrightnessContrast ─────────────────────────────────────

class BrightnessContrast(Transform):
    """
    Ajusta el brillo y el contraste de la imagen.
    No modifica los polígonos (transformación fotométrica pura).

    Fórmula: pixel_out = clip(pixel * contrast + brightness, 0, 255)
    """

    def __init__(
        self,
        probability: float    = 1.0,
        brightness_delta: int = 30,
        contrast_min: float   = 0.8,
        contrast_max: float   = 1.2,
        **kwargs,
    ):
        super().__init__(probability)
        self.brightness_delta = brightness_delta
        self.contrast_min     = contrast_min
        self.contrast_max     = contrast_max

    def apply(
        self,
        image: np.ndarray,
        polygons: List[Polygon],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        brightness = int(rng.integers(-self.brightness_delta, self.brightness_delta + 1))
        contrast   = float(rng.uniform(self.contrast_min, self.contrast_max))

        img_float  = image.astype(np.float32)
        img_out    = np.clip(img_float * contrast + brightness, 0, 255).astype(np.uint8)

        return img_out, polygons, {
            "brightness": brightness,
            "contrast":   round(contrast, 4),
        }
