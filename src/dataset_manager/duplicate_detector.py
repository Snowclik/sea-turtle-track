"""
src/dataset_manager/duplicate_detector.py
Identifica imágenes duplicadas utilizando hashes perceptuales (pHash / dHash).
"""

from __future__ import annotations
import os
import glob
from PIL import Image

try:
    import imagehash
except ImportError:
    imagehash = None

class DuplicateDetector:
    def __init__(self, existing_images_dir: str, threshold: int = 5):
        """
        threshold: distancia Hamming máxima para considerar dos imágenes iguales.
        5 es un buen valor para detectar imágenes idénticas pero ligeramente comprimidas.
        """
        self.existing_dir = existing_images_dir
        self.threshold = threshold
        self.existing_hashes = {}
        
        if imagehash is not None:
            self._build_index()
            
    def _build_index(self):
        if not os.path.isdir(self.existing_dir):
            return
            
        imgs = glob.glob(os.path.join(self.existing_dir, "*.*"))
        valid_exts = [".jpg", ".jpeg", ".png"]
        imgs = [x for x in imgs if os.path.splitext(x)[1].lower() in valid_exts]
        
        for p in imgs:
            try:
                img = Image.open(p)
                # Compute dHash
                h = imagehash.dhash(img)
                self.existing_hashes[p] = h
            except Exception:
                pass

    def is_duplicate(self, new_image_path: str) -> bool:
        if imagehash is None:
            # Fallback a nombre de archivo si no hay librería
            basename = os.path.basename(new_image_path)
            return os.path.isfile(os.path.join(self.existing_dir, basename))
            
        try:
            img = Image.open(new_image_path)
            h = imagehash.dhash(img)
            
            for existing_path, existing_hash in self.existing_hashes.items():
                if h - existing_hash <= self.threshold:
                    return True
            return False
        except Exception:
            return True # Rechazar por error de lectura
