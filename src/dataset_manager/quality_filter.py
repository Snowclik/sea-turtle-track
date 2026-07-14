"""
src/dataset_manager/quality_filter.py
Evalúa que las imágenes entrantes cumplan con criterios mínimos (corrupción, tamaño).
"""

from __future__ import annotations
import os
from PIL import Image

class QualityFilter:
    def __init__(self, min_width: int = 128, min_height: int = 128):
        self.min_width = min_width
        self.min_height = min_height

    def is_valid(self, image_path: str) -> bool:
        if not os.path.isfile(image_path):
            return False
            
        # Comprobar extensión
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            return False
            
        try:
            img = Image.open(image_path)
            img.verify() # Comprobar corrupción de cabecera
            
            # Reabrir para chequear tamaño porque verify() cierra el pointer
            img = Image.open(image_path)
            w, h = img.size
            
            if w < self.min_width or h < self.min_height:
                return False
                
            return True
        except Exception:
            return False
