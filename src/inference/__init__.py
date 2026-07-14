"""
src/inference/__init__.py
Orquestador de validación externa.
"""

from __future__ import annotations
import os
import glob
from rich.console import Console

from .predictor import Predictor
from .visualizer import Visualizer
from .evaluator import Evaluator
from .exporter import Exporter

console = Console(highlight=False)

class ExternalValidator:
    def __init__(self, project_root: str):
        self.project_root = project_root
        
        self.images_dir = os.path.join(project_root, "dataset", "external_validation", "images")
        self.labels_dir = os.path.join(project_root, "dataset", "external_validation", "labels")
        self.reports_dir = os.path.join(project_root, "reports", "external_validation")
        self.preds_dir = os.path.join(self.reports_dir, "predictions")
        
        # Crear directorios para que no falle si el usuario no los creó
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
    def validate(self):
        console.print(f"\n[bold magenta]=====================================================[/bold magenta]")
        console.print(f"[bold magenta] INICIANDO VALIDACIÓN EXTERNA DEL OFFICIAL_MODEL [/bold magenta]")
        console.print(f"[bold magenta]=====================================================[/bold magenta]\n")
        
        # 1. Escanear dataset externo
        imgs = glob.glob(os.path.join(self.images_dir, "*.*"))
        valid_exts = [".jpg", ".jpeg", ".png"]
        imgs = [x for x in imgs if os.path.splitext(x)[1].lower() in valid_exts]
        
        if not imgs:
            console.print(f"[yellow]⚠ No se encontraron imágenes en {self.images_dir}[/yellow]")
            console.print("[yellow]Coloca imágenes sin etiquetar o un dataset real en esa carpeta y vuelve a ejecutar el comando.[/yellow]")
            return
            
        console.print(f"  [dim]>> Cargando Predictor con OFFICIAL_MODEL...[/dim]")
        predictor = Predictor(self.project_root)
        visualizer = Visualizer(self.preds_dir)
        evaluator = Evaluator()
        exporter = Exporter(self.reports_dir)
        
        if predictor.model is None:
            return
            
        console.print(f"  [dim]>> Procesando {len(imgs)} imágenes de validación externa...[/dim]")
        
        detections = []
        total_time = 0.0
        
        # 2. Inferencia y Visualización
        for img in imgs:
            det = predictor.predict(img)
            if not det: continue
                
            detections.append(det)
            total_time += det["inference_time_s"]
            
            # Guardar preview
            basename = os.path.basename(img)
            visualizer.save_prediction(basename, det["result_obj"])
            
        # 3. Evaluación
        metrics = evaluator.evaluate(detections)
        
        avg_time = (total_time / len(imgs)) if imgs else 0.0
        fps = round((1.0 / avg_time), 2) if avg_time > 0 else 0
        
        summary = {
            "model_path": predictor.model_path,
            "total_images": len(imgs),
            "total_time_s": total_time,
            "avg_inference_time_s": avg_time,
            "fps": fps,
            "metrics": metrics
        }
        
        # 4. Reportes
        exporter.export(summary)
        
        console.print("\n[bold green]✔ Validación externa finalizada exitosamente.[/bold green]")
        console.print(f"[bold green]Se han procesado {len(imgs)} imágenes a {fps} FPS promedio.[/bold green]")
