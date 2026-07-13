"""
src/training/model_factory.py
Encargado de instanciar y configurar el modelo YOLO26n.
(Esqueleto: NO descargar pesos, NO inicializar PyTorch).
"""

from __future__ import annotations

class ModelFactory:
    def __init__(self, architecture: str, device: str):
        self.architecture = architecture
        self.device = device
        self.model_instance = None
        
    def build(self):
        """Prepara la inicialización del modelo (Mock por ahora)."""
        # Aquí se instanciará YOLO(f"{self.architecture}.pt") en el futuro.
        self.model_instance = f"Mocked_{self.architecture}_Instance"
        return self.model_instance
