"""
src/benchmark/benchmark_runner.py
Escanea los experimentos del ASGF y sincroniza los resultados al Leaderboard.
"""

from __future__ import annotations
import os
import glob
import json
from rich.console import Console
from rich.progress import track

from .leaderboard import BenchmarkLeaderboard
from .benchmark_registry import BenchmarkRegistry

console = Console(force_terminal=True, highlight=False)

class BenchmarkRunner:
    def __init__(self, experiments_dir: str, leaderboard: BenchmarkLeaderboard):
        self.experiments_dir = experiments_dir
        self.leaderboard = leaderboard

    def sync(self) -> int:
        """Sincroniza todos los experimentos encontrados al leaderboard."""
        pattern = os.path.join(self.experiments_dir, "exp_*")
        exp_dirs = glob.glob(pattern)
        
        count = 0
        for exp_dir in track(exp_dirs, description="Sincronizando experimentos..."):
            exp_id = os.path.basename(exp_dir)
            val_summary_path = os.path.join(exp_dir, "validation", "validation_summary.json")
            exp_summary_path = os.path.join(exp_dir, "experiment_summary.json")
            
            if not os.path.isfile(val_summary_path) or not os.path.isfile(exp_summary_path):
                continue
                
            with open(val_summary_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
                
            with open(exp_summary_path, "r", encoding="utf-8") as f:
                exp_data = json.load(f)
                
            # Combinar datos necesarios
            combined = val_data.copy()
            combined["config_name"] = exp_data.get("config_file", "unknown")
            combined["created_at"] = exp_data.get("created_at", "unknown")
            combined["config"] = exp_data.get("config", {})
            combined["base_dataset_fingerprint"] = exp_data.get("reproducibility", {}).get("global_seed", "unknown")
            
            # Cargar training info si existe
            train_summary = os.path.join(exp_dir, "training_results", "training_summary.json")
            if os.path.isfile(train_summary):
                with open(train_summary, "r", encoding="utf-8") as f:
                    combined["training_metrics"] = json.load(f)
                    
            # Cargar validation fingerprint si existe
            val_fp = os.path.join(exp_dir, "validation", "training_validation_fingerprint.json")
            combined["has_training_validation"] = os.path.isfile(val_fp)
            
            record = BenchmarkRegistry.create_record(exp_id, combined)
            self.leaderboard.add_or_update_experiment(exp_id, record)
            count += 1
            
        self.leaderboard.save()
        return count
