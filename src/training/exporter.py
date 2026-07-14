"""
src/training/exporter.py
Gestiona la exportación de los modelos entrenados.
(Esqueleto: NO exportar ONNX ni pesos aún).
"""

from __future__ import annotations

class ModelExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        
    def export(self, model_instance, yolo_run_dir: str):
        """Exporta el modelo entrenado a formato ONNX y copia los pesos."""
        import os
        import shutil
        from rich.console import Console
        console = Console(highlight=False)
        
        console.print("  [dim]>> Exportando modelo a formato ONNX...[/dim]")
        # Exportar a onnx
        try:
            onnx_path = model_instance.export(format="onnx")
            console.print(f"  [green]✔ ONNX exportado:[/green] {onnx_path}")
        except Exception as e:
            console.print(f"  [red]✖ Error exportando ONNX:[/red] {e}")
            
        # YOLO guarda los pesos en yolo_run_dir/weights/best.pt
        weights_dir = os.path.join(yolo_run_dir, "weights")
        best_pt = os.path.join(weights_dir, "best.pt")
        last_pt = os.path.join(weights_dir, "last.pt")
        best_onnx = os.path.join(weights_dir, "best.onnx")
        
        final_out = os.path.join(self.output_dir, "exported_models")
        os.makedirs(final_out, exist_ok=True)
        
        if os.path.isfile(best_pt): shutil.copy(best_pt, os.path.join(final_out, "best.pt"))
        if os.path.isfile(last_pt): shutil.copy(last_pt, os.path.join(final_out, "last.pt"))
        if os.path.isfile(best_onnx): shutil.copy(best_onnx, os.path.join(final_out, "best.onnx"))
