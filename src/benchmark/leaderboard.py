"""
src/benchmark/leaderboard.py
Maneja la lectura y escritura del archivo JSON central del Leaderboard.
"""

from __future__ import annotations
import os
import json
from datetime import datetime
from typing import Dict, Any

class BenchmarkLeaderboard:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.filepath = os.path.join(output_dir, "benchmark_leaderboard.json")
        os.makedirs(self.output_dir, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        """Carga el leaderboard existente o inicializa uno nuevo."""
        if os.path.isfile(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "last_updated": datetime.now().isoformat(timespec="seconds"),
            "experiments": {}
        }

    def save(self):
        """Guarda el leaderboard actualizándolo con la fecha actual."""
        self.data["last_updated"] = datetime.now().isoformat(timespec="seconds")
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)

    def add_or_update_experiment(self, exp_id: str, record: Dict[str, Any]):
        """Agrega o actualiza el registro de un experimento."""
        if exp_id not in self.data["experiments"]:
            self.data["experiments"][exp_id] = record
        else:
            # Hacemos merge de metadata y dataset_stats. No sobreescribir training_metrics si ya existen
            existing = self.data["experiments"][exp_id]
            existing["metadata"].update(record.get("metadata", {}))
            existing["dataset_stats"].update(record.get("dataset_stats", {}))
            if "training_metrics" in record:
                if "training_metrics" not in existing:
                    existing["training_metrics"] = {}
                existing["training_metrics"].update(record["training_metrics"])

    def get_all(self) -> Dict[str, Any]:
        return self.data["experiments"]
