"""
src/synthetic_generation/validator.py
─────────────────────────────────────
Única responsabilidad: validar que los polígonos de segmentación
resultantes de una transformación sean geométricamente válidos.

Reglas:
  - Mínimo de vértices por polígono: 3.
  - Todas las coordenadas deben estar en el rango [0, 1].
  - El área del polígono debe ser > 0 (no degenerado).
  - No se corrige nada automáticamente; solo se informa.
"""

from __future__ import annotations
from typing import List, Tuple

# Un polígono es una lista de (x, y) normalizados
Polygon = List[Tuple[float, float]]


class PolygonValidationError(Exception):
    """Excepción lanzada cuando un polígono no supera la validación."""
    pass


def validate_polygon(polygon: Polygon, index: int = 0) -> List[str]:
    """
    Valida un único polígono.
    Devuelve lista de issues (vacía = válido).
    """
    issues: List[str] = []

    # Vértices mínimos
    if len(polygon) < 3:
        issues.append(f"Polygon[{index}]: menos de 3 vértices ({len(polygon)})")
        return issues  # sin más verificaciones

    # Coordenadas fuera de [0, 1]
    for j, (x, y) in enumerate(polygon):
        if not (0.0 <= x <= 1.0):
            issues.append(f"Polygon[{index}] vertex[{j}]: x={x:.6f} fuera de [0,1]")
        if not (0.0 <= y <= 1.0):
            issues.append(f"Polygon[{index}] vertex[{j}]: y={y:.6f} fuera de [0,1]")

    # Área (Shoelace) > 0
    area = _shoelace_area(polygon)
    if area == 0.0:
        issues.append(f"Polygon[{index}]: área = 0 (polígono degenerado)")

    return issues


def validate_polygons(polygons: List[Polygon]) -> Tuple[bool, List[str]]:
    """
    Valida todos los polígonos de una imagen.
    Devuelve (es_válido, lista_de_issues).
    """
    all_issues: List[str] = []
    for i, poly in enumerate(polygons):
        all_issues.extend(validate_polygon(poly, index=i))
    return (len(all_issues) == 0, all_issues)


def clip_polygon(polygon: Polygon) -> Polygon:
    """
    Recorta las coordenadas al rango [0, 1].
    Útil como paso de limpieza tras transformaciones con pequeños errores numéricos.
    No altera la geometría; solo corrige valores marginalmente fuera de rango.
    """
    return [(min(1.0, max(0.0, x)), min(1.0, max(0.0, y))) for x, y in polygon]


def _shoelace_area(polygon: Polygon) -> float:
    """Área signed del polígono (normalizado). Solo el módulo importa."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
