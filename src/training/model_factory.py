"""
src/training/model_factory.py
Encargado de instanciar y configurar el modelo YOLO26n.
(Esqueleto: NO descargar pesos, NO inicializar PyTorch).
"""

from __future__ import annotations
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ModelFactory:
    def __init__(self, architecture: str, device: str):
        self.architecture = architecture
        self.device = device
        self.model_instance = None
        
    def build(self):
        """Prepara la inicialización del modelo YOLO."""
        if YOLO is None:
            raise ImportError("Ultralytics no está instalado.")
            
        model_file = f"{self.architecture}.pt"
        self.model_instance = YOLO(model_file)
        return self.model_instance
