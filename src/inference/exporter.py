"""
src/inference/exporter.py
Genera reportes multi-formato de la validación externa.
"""

from __future__ import annotations
import os
import json
from rich.console import Console

console = Console(highlight=False)

class Exporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export(self, summary_data: dict):
        self._export_json(summary_data)
        self._export_html(summary_data)
        console.print(f"  [green]✔ Reportes de validación externa generados en {self.output_dir}[/green]")
        
    def _export_json(self, data: dict):
        out = os.path.join(self.output_dir, "external_validation_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
    def _export_html(self, data: dict):
        out = os.path.join(self.output_dir, "external_validation_report.html")
        html = f"""<!DOCTYPE html><html><head><title>External Validation Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 50%; margin-top: 20px; }}
            th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style></head>
        <body>
            <h1>External Validation Report</h1>
            <p><strong>Modelo Oficial:</strong> {data.get("model_path", "unknown")}</p>
            <table>
                <tr><th>Métrica</th><th>Valor</th></tr>
                <tr><td>Imágenes Procesadas</td><td>{data.get("total_images", 0)}</td></tr>
                <tr><td>Objetos Detectados</td><td>{data.get("metrics", {{}}).get("total_detections", 0)}</td></tr>
                <tr><td>Confianza Promedio</td><td>{data.get("metrics", {{}}).get("avg_confidence", 0.0)}</td></tr>
                <tr><td>Tiempo Promedio Inferencia (ms)</td><td>{round(data.get("avg_inference_time_s", 0) * 1000, 2)}</td></tr>
                <tr><td>FPS (Aproximado)</td><td>{data.get("fps", 0)}</td></tr>
            </table>
            
            <br>
            <p>Revisar visualizaciones superpuestas en el directorio: <code>predictions/</code></p>
        </body></html>
        """
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
