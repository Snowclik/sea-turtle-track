"""
src/inference/visualizer.py
Genera y guarda las imágenes resultantes con máscaras, bboxes y confianzas superpuestas.
"""

from __future__ import annotations
import os
from rich.console import Console

console = Console(highlight=False)

class Visualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_prediction(self, img_filename: str, result_obj) -> str:
        out_path = os.path.join(self.output_dir, img_filename)
        
        try:
            # Ultralytics Results.save() automáticamente superpone bboxes, máscaras y confidencias
            result_obj.save(filename=out_path)
            return out_path
        except Exception as e:
            console.print(f"[red]⚠ Error guardando visualización de {img_filename}: {e}[/red]")
            return ""
