"""
src/training/adaptive_trainer.py
───────────────────────────────────
Orquestador de Entrenamiento Incremental (Adaptive Growth).
Encargado de ejecutar ventanas cortas de entrenamiento de YOLO sobre el dataset acumulativo,
encadenando los pesos (fine-tuning) para evitar Catastrophic Forgetting y ahorrar cómputo.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any

from ultralytics import YOLO
from rich.console import Console

console = Console()

class AdaptiveTrainer:
    def __init__(self, output_dir: str, dataset_yaml: str, base_model: str = "yolo26n-seg.pt"):
        self.output_dir = output_dir
        self.dataset_yaml = dataset_yaml
        self.base_model = base_model
        
        # Historial de pesos
        self.best_weights_path = None
        self.last_weights_path = None
        
        # Garantizar que el directorio de entrenamiento exista
        self.train_dir = os.path.join(self.output_dir, "adaptive_training")
        os.makedirs(self.train_dir, exist_ok=True)

    def train_step(self, iteration: int, epochs: int = 5, batch: int = 16, imgsz: int = 640) -> Dict[str, Any]:
        """
        Ejecuta una ventana de entrenamiento incremental.
        """
        console.print(f"\n[bold green]>> Inciando Entrenamiento Incremental — Iteración {iteration}[/bold green]")
        
        # 1. Determinar el modelo de origen
        if iteration == 0 or self.last_weights_path is None:
            model_path = self.base_model
            console.print(f"  [dim]Origen de pesos: Modelo Base ({model_path})[/dim]")
        else:
            model_path = self.last_weights_path
            console.print(f"  [dim]Origen de pesos: Iteración anterior ({model_path})[/dim]")
            
        model = YOLO(model_path)
        
        # 2. Configurar Argumentos (Bloqueando Aumentaciones de YOLO)
        run_name = f"iter_{iteration:03d}"
        
        train_args = {
            "data": self.dataset_yaml,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": "cpu",
            "workers": 4,
            "cache": True,           # RAM caching para velocidad I/O
            "project": self.train_dir,
            "name": run_name,
            "exist_ok": True,
            "plots": False,          # Ahorro masivo de tiempo en CPU (evitar graficación)
            "verbose": False,        # Output limpio
            
            # Bloqueo total de aumentaciones nativas (ASGF hace este trabajo)
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "erasing": 0.0
        }
        
        # 3. Entrenar
        start_time = time.time()
        results = model.train(**train_args)
        elapsed_time = time.time() - start_time
        
        # 4. Actualizar estado
        run_output_dir = os.path.join(self.train_dir, run_name)
        self.best_weights_path = os.path.join(run_output_dir, "weights", "best.pt")
        self.last_weights_path = os.path.join(run_output_dir, "weights", "last.pt")
        
        # 5. Extraer métricas clave
        res_dict = results.results_dict if hasattr(results, 'results_dict') else {}
        
        map50 = results.box.map50 if hasattr(results, 'box') else res_dict.get('metrics/mAP50(B)', 0.0)
        map50_95 = results.box.map if hasattr(results, 'box') else res_dict.get('metrics/mAP50-95(B)', 0.0)
        
        # Segmentación
        if hasattr(results, 'seg') and results.seg is not None:
            map50_95 = results.seg.map
            map50 = results.seg.map50
            precision = res_dict.get('metrics/precision(M)', res_dict.get('metrics/precision(B)', 0.0))
            recall = res_dict.get('metrics/recall(M)', res_dict.get('metrics/recall(B)', 0.0))
        else:
            precision = res_dict.get('metrics/precision(B)', 0.0)
            recall = res_dict.get('metrics/recall(B)', 0.0)
            
        # Calcular F1 Score (Harmonic mean of precision and recall)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        # Calcular Score Compuesto
        score = (0.45 * map50_95) + (0.30 * f1) + (0.15 * recall) + (0.10 * precision)
        
        metrics = {
            "iteration": iteration,
            "elapsed_seconds": round(elapsed_time, 2),
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "score": score,
            "best_pt": self.best_weights_path
        }
        
        console.print(f"  [cyan]Resultados Iteración {iteration}[/cyan]: Score={score:.4f} | mAP={map50_95:.4f} | F1={f1:.4f} | Tiempo={metrics['elapsed_seconds']}s")
        return metrics
