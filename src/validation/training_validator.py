"""
src/validation/training_validator.py
Auditor principal del entrenamiento que extrae métricas y consolida gráficas.
"""

from __future__ import annotations
import os
import shutil
import json
import hashlib
from datetime import datetime
from rich.console import Console

from .inference_tester import InferenceTester

console = Console(highlight=False)

class TrainingValidator:
    def __init__(self, exp_id: str, project_root: str):
        self.exp_id = exp_id
        self.project_root = project_root
        
        # Resolver el ID (ej. baseline -> exp_...)
        import sys
        if os.path.join(project_root, "src") not in sys.path:
            sys.path.insert(0, os.path.join(project_root, "src"))
        from training.dataset_loader import DatasetLoader
        
        ds = DatasetLoader(exp_id, project_root)
        self.exp_path = ds.experiment_path
        self.exp_id = os.path.basename(self.exp_path)
        
        self.yolo_run_dir = os.path.join(self.exp_path, "training_results", "yolo_run")
        self.graphs_dir = os.path.join(self.exp_path, "validation", "graphs")
        self.val_dir = os.path.join(self.exp_path, "validation")
        
    def validate(self):
        console.print(f"\n[bold cyan]Iniciando Validación Científica (Etapa 7.5): {self.exp_id}[/bold cyan]")
        
        if not os.path.isdir(self.yolo_run_dir):
            console.print("[red]✖ No se encontró la carpeta de resultados de entrenamiento de YOLO.[/red]")
            return False
            
        os.makedirs(self.graphs_dir, exist_ok=True)
        
        # 1. Extraer gráficas oficiales
        console.print("  [dim]>> Extrayendo gráficas de entrenamiento...[/dim]")
        self._copy_graphs()
        
        # 2. Inferencia y Comparación (PT vs ONNX)
        console.print("  [dim]>> Ejecutando auditoría de inferencia (PT vs ONNX)...[/dim]")
        tester = InferenceTester(self.exp_path, self.project_root)
        inf_results = tester.run_tests()
        
        # 3. Generar Fingerprint
        console.print("  [dim]>> Sellando validación...[/dim]")
        self._generate_fingerprint(inf_results)
        
        console.print(f"  [green]✔ Auditoría completada. Fingerprint sellado.[/green]\n")
        return True
        
    def _copy_graphs(self):
        targets = [
            "results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
            "labels.jpg", "labels_correlogram.jpg", 
            "MaskPR_curve.png", "MaskP_curve.png", "MaskR_curve.png", "MaskF1_curve.png",
            "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png", "BoxF1_curve.png"
        ]
        
        count = 0
        for f in targets:
            src = os.path.join(self.yolo_run_dir, f)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(self.graphs_dir, f))
                count += 1
                
        console.print(f"  [green]✔ {count} gráficas/recursos extraídos a validation/graphs.[/green]")
        
    def _generate_fingerprint(self, inf_results: dict):
        # Hashear pesos
        weights_dir = os.path.join(self.exp_path, "training_results", "exported_models")
        pt_path = os.path.join(weights_dir, "best.pt")
        onnx_path = os.path.join(weights_dir, "best.onnx")
        
        def md5(fname):
            if not os.path.isfile(fname): return "N/A"
            hash_md5 = hashlib.md5()
            with open(fname, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
            
        fp = {
            "model": "YOLO26n-Seg",
            "date": datetime.now().isoformat(),
            "experiment_id": self.exp_id,
            "hash_best_pt": md5(pt_path),
            "hash_best_onnx": md5(onnx_path),
            "inference_results": inf_results
        }
        
        out = os.path.join(self.val_dir, "training_validation_fingerprint.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(fp, f, indent=4)
