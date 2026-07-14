"""
src/training/tuner.py
Implementa la lógica de Hyperparameter Tuning (Evolución Genética) para YOLO26.
"""

from __future__ import annotations
import os
import sys
from rich.console import Console

from .dataset_loader import DatasetLoader
from .protocol_manager import TrainingProtocolAdapter
from .model_factory import ModelFactory

console = Console(highlight=False)

class YoloTuner:
    def __init__(self, experiment_id: str, project_root: str):
        self.experiment_id = experiment_id
        self.project_root = project_root
        
    def tune(self, iterations: int = 30):
        console.print(f"\n[bold magenta]=====================================================[/bold magenta]")
        console.print(f"[bold magenta] INICIANDO HYPERPARAMETER TUNING: {self.experiment_id} [/bold magenta]")
        console.print(f"[bold magenta]=====================================================[/bold magenta]\n")
        
        console.print("  [dim]>> Resolviendo dataset...[/dim]")
        ds = DatasetLoader(self.experiment_id, self.project_root)
        ds_info = ds.prepare_for_training()
        
        console.print("  [dim]>> Cargando Protocolo Experimental...[/dim]")
        protocol = TrainingProtocolAdapter()
        cfg = protocol.get_config()
        
        arch = cfg.get("model", {}).get("architecture", "unknown")
        device = cfg.get("hardware", {}).get("device", "auto")
        if device == "auto":
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
            
        factory = ModelFactory(architecture=arch, device=device)
        model = factory.build()
        
        tune_out = os.path.join(ds_info["experiment_path"], "tuning_results")
        os.makedirs(tune_out, exist_ok=True)
        
        console.print(f"  [bold yellow]>> Mutando hiperparámetros (Iteraciones: {iterations})[/bold yellow]")
        console.print(f"  [dim]Este proceso usará algoritmos genéticos y tardará varias horas.[/dim]")
        
        tune_args = {
            "data": ds_info["data_yaml"],
            "epochs": min(cfg.get("training", {}).get("epochs", 30), 50), # Normalmente el tuning se hace a menos épocas
            "iterations": iterations,
            "imgsz": cfg.get("training", {}).get("imgsz", 640),
            "batch": cfg.get("training", {}).get("batch", 16),
            "device": device,
            "project": tune_out,
            "name": "yolo_tune",
            "exist_ok": True,
            "optimizer": cfg.get("training", {}).get("optimizer", "auto")
        }
        
        model.tune(**tune_args)
        
        console.print(f"\n[bold green]✔ Tuning completado.[/bold green]")
        console.print(f"[bold green]Mejores hiperparámetros guardados en {tune_out}/yolo_tune/best_hyperparameters.yaml[/bold green]")
