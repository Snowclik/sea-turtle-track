"""
src/inference/predictor.py
Se encarga de cargar el OFFICIAL_MODEL y realizar inferencias sobre imágenes.
"""

from __future__ import annotations
import os
import time
from rich.console import Console

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

console = Console(highlight=False)

class Predictor:
    def __init__(self, project_root: str):
        self.project_root = project_root
        # Preferir ONNX por velocidad, fallback a PT
        onnx_path = os.path.join(project_root, "OFFICIAL_MODEL", "yolo26n-seg_asgf_official.onnx")
        pt_path = os.path.join(project_root, "OFFICIAL_MODEL", "yolo26n-seg_asgf_official.pt")
        
        self.model_path = onnx_path if os.path.isfile(onnx_path) else pt_path
        self.model = None
        
        if not os.path.isfile(self.model_path):
            console.print(f"[red]✖ No se encontró el OFFICIAL_MODEL en {self.model_path}[/red]")
        elif YOLO is not None:
            self.model = YOLO(self.model_path, task='segment')
            
    def predict(self, img_path: str) -> dict:
        if self.model is None:
            return {}
            
        t0 = time.time()
        # verbose=False para no ensuciar la consola
        results = self.model(img_path, verbose=False)
        t1 = time.time()
        
        res = results[0]
        
        return {
            "result_obj": res,
            "inference_time_s": t1 - t0,
            "num_detections": len(res.boxes) if res.boxes else 0
        }
