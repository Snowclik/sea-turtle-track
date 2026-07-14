"""
src/benchmark/benchmark_registry.py
Lógica de estado y validación para el registro de un experimento.
Convierte el output del ASGF en un registro normalizado para el Leaderboard.
"""

from __future__ import annotations
from typing import Dict, Any

from .metrics import TrainingMetrics

class BenchmarkRegistry:
    @staticmethod
    def create_record(exp_id: str, summary: dict) -> Dict[str, Any]:
        """Convierte un validation_summary.json + experiment_summary.json a un record del Leaderboard."""
        
        status = "DATASET_READY"
        if "training_metrics" in summary:
            status = "TRAINED"
            if summary.get("has_training_validation"):
                status = "VALIDATED"
                
        metadata = {
            "config_name": summary.get("config_name", "unknown"),
            "dataset_fingerprint": summary.get("base_dataset_fingerprint", "unknown"),
            "pipeline_fingerprint": summary.get("pipeline_fingerprint", "unknown"),
            "date": summary.get("created_at", "unknown"),
            "status": status
        }
        
        transforms = []
        if "config" in summary and "transforms" in summary["config"]:
            transforms = [k for k, v in summary["config"]["transforms"].items() if v.get("enabled", False)]
            
        dataset_stats = {
            "num_images": summary.get("total_generated", 0),
            "num_objects": 0, # Reservado para cuando contemos todos los polígonos
            "generation_time_s": summary.get("elapsed_seconds", 0.0),
            "transforms_applied": transforms
        }
        
        training_metrics = summary.get("training_metrics", TrainingMetrics().to_dict())
        
        return {
            "metadata": metadata,
            "dataset_stats": dataset_stats,
            "training_metrics": training_metrics
        }
