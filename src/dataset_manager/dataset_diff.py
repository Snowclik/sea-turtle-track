"""
src/dataset_manager/dataset_diff.py
Compara el estado físico actual del dataset original contra el último registro oficial.
"""

from __future__ import annotations
import os
import glob
from rich.console import Console

from .dataset_versioning import DatasetVersioner

console = Console(highlight=False)

class DatasetDiff:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.dataset_dir = os.path.join(project_root, "dataset", "original")
        self.versioner = DatasetVersioner(project_root)
        
    def diff(self):
        console.print(f"\n[bold magenta]=====================================================[/bold magenta]")
        console.print(f"[bold magenta] DATASET EXPANSION FRAMEWORK: DIFF [/bold magenta]")
        console.print(f"[bold magenta]=====================================================[/bold magenta]\n")
        
        # Estado actual físico
        imgs = glob.glob(os.path.join(self.dataset_dir, "*.jpg")) + glob.glob(os.path.join(self.dataset_dir, "*.png"))
        txts = glob.glob(os.path.join(self.dataset_dir, "*.txt"))
        
        current_images = len(imgs)
        # Contar objetos en TXT
        current_objects = 0
        for txt in txts:
            try:
                with open(txt, "r", encoding="utf-8") as f:
                    lines = [l for l in f.read().splitlines() if l.strip()]
                    current_objects += len(lines)
            except:
                pass
                
        # Estado anterior
        last_stats = self.versioner.get_latest_stats()
        last_images = last_stats.get("images", 0)
        last_objects = last_stats.get("objects", 0)
        
        delta_img = current_images - last_images
        delta_obj = current_objects - last_objects
        
        # Presentar diff
        if delta_img == 0 and delta_obj == 0 and last_images > 0:
            console.print("[dim]El dataset no presenta modificaciones respecto a la versión anterior.[/dim]")
            return False
            
        console.print("[cyan]Cambios detectados:[/cyan]")
        sign_img = "+" if delta_img > 0 else ""
        sign_obj = "+" if delta_obj > 0 else ""
        
        console.print(f"  {sign_img}{delta_img} imágenes nuevas (Total: {current_images})")
        console.print(f"  {sign_obj}{delta_obj} objetos nuevos (Total: {current_objects})")
        
        if last_images == 0:
            console.print("\n[dim]* Esta será la versión inicial (v1).[/dim]")
            
        return {"current_img": current_images, "current_obj": current_objects}
