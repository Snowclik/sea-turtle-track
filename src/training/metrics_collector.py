"""
src/training/metrics_collector.py
Recolecta y formatea las métricas del entrenamiento para el Benchmark.
(Esqueleto).
"""

from __future__ import annotations

class MetricsCollector:
    """Prepara la estructura de resultados compatible con el Leaderboard."""
    
    def __init__(self):
        self.metrics = {
            "mAP_50": None,
            "mAP_50_95": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "training_time_s": None,
            "inference_time_ms": None,
            "model_size_mb": None,
            "params_m": None
        }
        
    def collect(self) -> dict:
        """Devuelve el diccionario final de métricas."""
        # En el futuro, leerá de results.csv o del objeto YOLO
        return self.metrics
