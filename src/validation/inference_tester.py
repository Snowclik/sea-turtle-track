"""
src/validation/inference_tester.py
Módulo dedicado a realizar inferencias comparativas entre el modelo PyTorch y el exportado a ONNX.
"""

from __future__ import annotations
import os
import time
import glob
import random
from rich.console import Console

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

console = Console(highlight=False)

class InferenceTester:
    def __init__(self, exp_path: str, project_root: str):
        self.exp_path = exp_path
        self.project_root = project_root
        self.out_dir = os.path.join(exp_path, "validation", "inference")
        os.makedirs(self.out_dir, exist_ok=True)
        
        weights_dir = os.path.join(exp_path, "training_results", "exported_models")
        self.pt_path = os.path.join(weights_dir, "best.pt")
        self.onnx_path = os.path.join(weights_dir, "best.onnx")

    def run_tests(self) -> dict:
        console.print("  [dim]>> Seleccionando imágenes de prueba...[/dim]")
        # Seleccionar 3 imágenes originales
        orig_imgs = glob.glob(os.path.join(self.project_root, "dataset", "original", "*.jpg"))
        orig_imgs += glob.glob(os.path.join(self.project_root, "dataset", "original", "*.png"))
        orig_imgs += glob.glob(os.path.join(self.project_root, "dataset", "original", "*.jpeg"))
        
        if not orig_imgs:
            console.print("  [red]✖ No se encontraron imágenes originales para validar.[/red]")
            return {}
            
        test_imgs = random.sample(orig_imgs, min(3, len(orig_imgs)))
        
        if YOLO is None:
            console.print("  [red]✖ Ultralytics no instalado, cancelando inferencia.[/red]")
            return {}

        results = {
            "pytorch": self._run_model(self.pt_path, test_imgs, "pt"),
            "onnx": self._run_model(self.onnx_path, test_imgs, "onnx")
        }
        
        return results

    def _run_model(self, model_path: str, imgs: list, suffix: str) -> dict:
        if not os.path.isfile(model_path):
            console.print(f"  [yellow]⚠ Modelo {suffix} no encontrado en {model_path}[/yellow]")
            return {}
            
        model = YOLO(model_path, task='segment')
        
        total_time = 0
        total_dets = 0
        avg_conf = 0.0
        
        for i, img_path in enumerate(imgs):
            t0 = time.time()
            res = model(img_path, verbose=False)
            t1 = time.time()
            
            inf_time = t1 - t0
            total_time += inf_time
            
            r = res[0]
            num_dets = len(r.boxes) if r.boxes else 0
            total_dets += num_dets
            
            if num_dets > 0:
                confs = r.boxes.conf.cpu().numpy()
                avg_conf += sum(confs)
                
            # Guardar preview
            out_img = os.path.join(self.out_dir, f"test_{i}_{suffix}.jpg")
            r.save(filename=out_img)
            
        final_conf = avg_conf / total_dets if total_dets > 0 else 0.0
        
        return {
            "avg_inference_time_s": round(total_time / len(imgs), 4),
            "total_detections": total_dets,
            "avg_confidence": round(float(final_conf), 4)
        }
