"""
src/synthetic_generation/pipeline.py
─────────────────────────────────────
Única responsabilidad: definir el flujo de pasos por imagen.

Dado una imagen de origen + sus polígonos, el pipeline ejecuta:
  1. Copiar imagen y polígonos para no modificar los originales.
  2. Para cada transformación habilitada: aplicar y registrar.
  3. Clip de coordenadas a [0, 1] para corregir errores numéricos marginales.
  4. Validar polígonos resultantes.
  5. Devolver resultado (imagen, polígonos, records, valid).

El pipeline NO escribe archivos. Eso es responsabilidad del Exporter.
"""

from __future__ import annotations
import copy
from typing import List, Tuple, Dict, Any

import numpy as np

from transforms import Transform
from validator import validate_polygons, clip_polygon
from metadata import TransformRecord

Polygon = List[Tuple[float, float]]


class SyntheticPipeline:
    """
    Orquesta la secuencia de transformaciones sobre una imagen.
    Inmutable: el mismo pipeline puede usarse para múltiples imágenes.
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms

    def run(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        rng:      np.random.Generator,
    ) -> Tuple[np.ndarray, List[Polygon], List[TransformRecord], bool, List[str]]:
        """
        Ejecuta el pipeline sobre una imagen.

        Retorna:
            img_out      — imagen resultante
            polys_out    — polígonos resultantes
            records      — lista de TransformRecord (para trazabilidad)
            is_valid     — True si todos los polígonos son válidos
            issues       — lista de problemas encontrados
        """
        img_out   = image.copy()
        polys_out = copy.deepcopy(polygons)
        records:  List[TransformRecord] = []

        for transform in self.transforms:
            img_out, polys_out, params, applied = transform(img_out, polys_out, rng)
            records.append(TransformRecord(
                name=transform.name,
                parameters=params,
                applied=applied,
            ))

        # Clip de coordenadas marginalmente fuera de [0, 1]
        polys_out = [clip_polygon(poly) for poly in polys_out]

        # Validar polígonos resultantes
        is_valid, issues = validate_polygons(polys_out)

        return img_out, polys_out, records, is_valid, issues
