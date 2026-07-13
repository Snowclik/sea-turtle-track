"""
src/synthetic_generation/verifier/visualizer.py
────────────────────────────────────────────────
Selecciona aleatoriamente imágenes comparativas y las guarda en validation/visual_samples/
"""

from __future__ import annotations
import os
import glob
import random
import shutil

class Visualizer:
    def __init__(self, experiment_dir: str, num_samples: int = 10):
        self.experiment_dir = experiment_dir
        self.num_samples = num_samples
        self.previews_dir = os.path.join(experiment_dir, "previews")
        self.out_dir = os.path.join(experiment_dir, "validation", "visual_samples")

    def generate_samples(self) -> list[str]:
        """Copia `num_samples` previews aleatorios a validation/visual_samples/"""
        os.makedirs(self.out_dir, exist_ok=True)
        
        previews = sorted(glob.glob(os.path.join(self.previews_dir, "*.jpg")))
        if not previews:
            return []

        # Seleccionar aleatorios reproducibles
        random.seed(42)
        samples = random.sample(previews, min(self.num_samples, len(previews)))

        copied = []
        for i, src in enumerate(samples, 1):
            name = f"sample_{i:04d}_{os.path.basename(src)}"
            dst = os.path.join(self.out_dir, name)
            shutil.copy2(src, dst)
            copied.append(dst)
            
        return copied
