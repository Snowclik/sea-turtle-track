"""
src/training/exporter.py
Gestiona la exportación de los modelos entrenados.
(Esqueleto: NO exportar ONNX ni pesos aún).
"""

from __future__ import annotations

class ModelExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        
    def export(self, model_instance):
        """Prepara la lógica para exportar best.pt, last.pt y best.onnx."""
        pass
