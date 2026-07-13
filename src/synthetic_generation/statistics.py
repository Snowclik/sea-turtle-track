"""
src/synthetic_generation/statistics.py
─────────────────────────────────────
Única responsabilidad: acumular y reportar las estadísticas
de un run completo del motor de generación sintética.
"""

from __future__ import annotations
import json
import os
import time
from collections import defaultdict
from typing import Dict, Any


class RunStatistics:
    """Acumula estadísticas durante un run del motor."""

    def __init__(self, config_path: str = ""):
        self._start_time   = time.time()
        self.config_path   = config_path
        self.requested     = 0      # imágenes solicitadas
        self.generated     = 0      # imágenes generadas exitosamente
        self.discarded     = 0      # imágenes descartadas (polígono inválido)
        self.errors        = 0      # errores inesperados
        self.retries       = 0      # reintentos totales
        self.transform_counts: Dict[str, int] = defaultdict(int)  # nombre → veces aplicada
        self.elapsed_seconds = 0.0

    # ── Actualizadores ────────────────────────────────────────────────────────

    def record_generated(self) -> None:
        self.generated += 1

    def record_discarded(self, reason: str = "") -> None:
        self.discarded += 1

    def record_error(self) -> None:
        self.errors += 1

    def record_retry(self) -> None:
        self.retries += 1

    def record_transform(self, name: str) -> None:
        self.transform_counts[name] += 1

    # ── Cierre ────────────────────────────────────────────────────────────────

    def finish(self) -> None:
        self.elapsed_seconds = round(time.time() - self._start_time, 2)

    # ── Serialización ─────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path":       self.config_path,
            "requested":         self.requested,
            "generated":         self.generated,
            "discarded":         self.discarded,
            "errors":            self.errors,
            "retries":           self.retries,
            "success_rate_pct":  round((self.generated / self.requested) * 100, 2)
                                 if self.requested else 0.0,
            "elapsed_seconds":   self.elapsed_seconds,
            "transforms_applied": dict(self.transform_counts),
        }

    def save(self, output_path: str) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
