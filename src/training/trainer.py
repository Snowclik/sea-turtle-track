"""
src/training/trainer.py
Esqueleto de la clase que envolverá a YOLO26.
No implementa el entrenamiento todavía.
"""

from __future__ import annotations
import os
import time

from .metrics_collector import MetricsCollector
from .exporter import ModelExporter
from .callbacks import TrainingCallbacks

class YoloTrainer:
    def __init__(self, model_instance, protocol_cfg: dict, dataset_info: dict, output_dir: str):
        self.model = model_instance
        self.cfg = protocol_cfg
        self.dataset_info = dataset_info
        self.output_dir = output_dir
        
        self.metrics = MetricsCollector()
        self.exporter = ModelExporter(output_dir)
        self.callbacks = TrainingCallbacks()

    def train(self, resume: bool = False):
        """Ejecuta el entrenamiento local de YOLO26."""
        self.callbacks.on_train_start()
        
        # Iniciar entrenamiento
        # Parámetros estrictos desde el Protocolo Experimental
        device = self.cfg.get("hardware", {}).get("device", "auto")
        if device == "auto":
            import torch
            device = "0" if torch.cuda.is_available() else "cpu"
            
        train_args = {
            "data": self.dataset_info["data_yaml"],
            "epochs": self.cfg.get("training", {}).get("epochs", 300),
            "imgsz": self.cfg.get("training", {}).get("imgsz", 640),
            "batch": self.cfg.get("training", {}).get("batch", 16),
            "optimizer": self.cfg.get("training", {}).get("optimizer", "auto"),
            "patience": self.cfg.get("training", {}).get("patience", 50),
            "seed": self.cfg.get("training", {}).get("seed", 42),
            "workers": self.cfg.get("training", {}).get("workers", 4),
            "device": device,
            "project": self.output_dir,
            "name": "yolo_run",
            "exist_ok": True,
            "verbose": False,
            "cache": True,        # Opt 2: Caché en RAM
            "plots": False        # Opt 4: Sin gráficas por época
        }
        
        # Inyectar configuraciones dinámicas de Data Augmentation de YOLO (Online)
        yolo_aug = self.cfg.get("yolo_augmentations", {})
        
        # FORZAR DESACTIVACIÓN DE AUGMENTATIONS INTERNAS (Prioridad 1)
        yolo_aug.update({
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
        })
        
        train_args.update(yolo_aug)
        
        # Manejo de resume
        if resume:
            last_pt = os.path.join(self.output_dir, "yolo_run", "weights", "last.pt")
            if os.path.isfile(last_pt):
                train_args["resume"] = True
                # YOLO auto-detects last.pt if we set resume=True and it's the same project/name, 
                # but to be safe we load it explicitly
                self.model = type(self.model)(last_pt)
            else:
                from rich.console import Console
                Console().print(f"[red]⚠ No se encontró {last_pt}. Iniciando desde cero.[/red]")
        
        import time
        start_t = time.time()
        self.model.train(**train_args)
        end_t = time.time()
        
        self.callbacks.on_train_end()
        
        # Exportación (ONNX, best, last)
        self.exporter.export(self.model, os.path.join(self.output_dir, "yolo_run"))
        
        # Colectar métricas
        return self.metrics.collect(
            yolo_run_dir=os.path.join(self.output_dir, "yolo_run"), 
            total_time=end_t - start_t,
            model_instance=self.model
        )
