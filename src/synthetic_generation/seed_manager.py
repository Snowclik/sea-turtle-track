"""
src/synthetic_generation/seed_manager.py
─────────────────────────────────────────
Única responsabilidad: centralizar toda la gestión de semillas
para garantizar reproducibilidad completa del motor ASDGE.

Principio:
  - Una semilla global define completamente una ejecución.
  - Cada imagen recibe una semilla derivada de forma determinista.
  - La misma semilla global + los mismos datos = los mismos resultados.
"""

from __future__ import annotations
import numpy as np


class SeedManager:
    """
    Gestor central de semillas para reproducibilidad.

    Uso:
        sm = SeedManager(global_seed=42)
        rng_img = sm.image_rng(image_index=7)   # determinista
    """

    def __init__(self, global_seed: int):
        if not (0 <= global_seed < 2**32):
            raise ValueError(f"global_seed debe estar en [0, 2^32), recibido: {global_seed}")
        self._global_seed  = global_seed
        self._global_rng   = np.random.default_rng(global_seed)
        self._image_seeds: list[int] = []   # historial de semillas por imagen

    @property
    def global_seed(self) -> int:
        return self._global_seed

    # ── Semillas por imagen ───────────────────────────────────────────────────

    def next_image_seed(self) -> int:
        """
        Genera la semilla para la siguiente imagen de forma determinista.
        El orden de llamadas importa: la misma secuencia siempre produce
        los mismos resultados con la misma semilla global.
        """
        seed = int(self._global_rng.integers(0, 2**32))
        self._image_seeds.append(seed)
        return seed

    def image_rng(self, seed: int) -> np.random.Generator:
        """Devuelve un RNG aislado para una imagen específica."""
        return np.random.default_rng(seed)

    # ── Selección reproducible de fuente ─────────────────────────────────────

    def select_source_index(self, n_sources: int) -> int:
        """Selecciona un índice de fuente de forma reproducible."""
        return int(self._global_rng.integers(0, n_sources))

    # ── Historial ─────────────────────────────────────────────────────────────

    def seed_history(self) -> list[int]:
        """Devuelve la lista de semillas por imagen generadas hasta ahora."""
        return list(self._image_seeds)

    def to_dict(self) -> dict:
        return {
            "global_seed":  self._global_seed,
            "image_seeds":  self._image_seeds,
        }
