"""
src/synthetic_generation/transforms/base.py
────────────────────────────────────────────
Única responsabilidad: definir el contrato abstracto que toda
transformación del motor ASDGE debe implementar.

Contrato:
  - Entrada:  imagen (np.ndarray), polígonos (List[Polygon]), rng (np.random.Generator)
  - Salida:   (imagen_transformada, polígonos_transformados, params_aplicados, applied)
  - NUNCA convierte polígonos a bboxes.
  - Siempre devuelve coordenadas normalizadas [0, 1].
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

import numpy as np

# Tipo alias compartido por todo el paquete
Polygon = List[Tuple[float, float]]


class Transform(ABC):
    """
    Clase base abstracta para todas las transformaciones del ASDGE.

    Subclases deben implementar únicamente `apply()`.
    La lógica de probabilidad está centralizada aquí.
    """

    def __init__(self, probability: float = 1.0, **kwargs):
        if not (0.0 <= probability <= 1.0):
            raise ValueError(
                f"[{self.__class__.__name__}] probability debe estar en [0, 1], "
                f"recibido: {probability}"
            )
        self.probability = probability

    def __call__(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any], bool]:
        """
        Evalúa la probabilidad y aplica la transformación si corresponde.

        Returns:
            img_out   — imagen resultante (original o transformada)
            polys_out — polígonos resultantes
            params    — parámetros aplicados (vacío si no se aplicó)
            applied   — True si la transformación fue ejecutada
        """
        if rng.random() <= self.probability:
            img_out, polys_out, params = self.apply(image, polygons, rng)
            return img_out, polys_out, params, True
        return image, polygons, {}, False

    @abstractmethod
    def apply(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], Dict[str, Any]]:
        """
        Aplica la transformación incondicionalmente.
        La evaluación de probabilidad ocurre en __call__.
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def category(self) -> str:
        """Categoría para clasificación: 'geometric' | 'photometric' | 'custom'."""
        return "base"

    def __repr__(self) -> str:
        return f"{self.name}(p={self.probability})"
