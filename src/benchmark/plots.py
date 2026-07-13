"""
src/benchmark/plots.py
(Esqueleto) Funciones para generar gráficos comparativos del benchmark.
"""

from __future__ import annotations
from typing import Dict, Any

class BenchmarkPlotter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def plot_training_metrics(self, data: Dict[str, Any]):
        """Generará gráficos cuando haya métricas de YOLO."""
        pass
