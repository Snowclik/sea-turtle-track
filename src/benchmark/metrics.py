"""
src/benchmark/metrics.py
Estructuras de datos para las métricas de entrenamiento.
Por ahora inicializadas en None.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TrainingMetrics:
    """Estructura de datos para las métricas de YOLO26."""
    mAP_50: Optional[float] = None
    mAP_50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Métricas de rendimiento computacional
    training_time_s: Optional[float] = None
    inference_time_ms: Optional[float] = None
    model_size_mb: Optional[float] = None
    params_m: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mAP_50": self.mAP_50,
            "mAP_50_95": self.mAP_50_95,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "training_time_s": self.training_time_s,
            "inference_time_ms": self.inference_time_ms,
            "model_size_mb": self.model_size_mb,
            "params_m": self.params_m
        }
