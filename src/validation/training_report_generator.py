"""
src/validation/training_report_generator.py
Exporta los resultados de la validación a PDF, HTML, JSON y XLSX.
"""

from __future__ import annotations
import os
import json

class TrainingReportGenerator:
    def __init__(self, exp_id: str, project_root: str):
        import sys
        if os.path.join(project_root, "src") not in sys.path:
            sys.path.insert(0, os.path.join(project_root, "src"))
        from training.dataset_loader import DatasetLoader
        
        ds = DatasetLoader(exp_id, project_root)
        self.exp_path = ds.experiment_path
        self.exp_id = os.path.basename(self.exp_path)
        
        self.out_dir = os.path.join(self.exp_path, "validation", "reports")
        os.makedirs(self.out_dir, exist_ok=True)
        
    def generate(self):
        fp_path = os.path.join(self.exp_path, "validation", "training_validation_fingerprint.json")
        train_sum = os.path.join(self.exp_path, "training_results", "training_summary.json")
        
        data = {}
        if os.path.isfile(fp_path):
            with open(fp_path, "r", encoding="utf-8") as f:
                data["validation"] = json.load(f)
                
        if os.path.isfile(train_sum):
            with open(train_sum, "r", encoding="utf-8") as f:
                data["training"] = json.load(f)
                
        self._export_json(data)
        self._export_html(data)
        
    def _export_json(self, data: dict):
        out = os.path.join(self.out_dir, "training_validation_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
            
    def _export_html(self, data: dict):
        out = os.path.join(self.out_dir, "training_validation_report.html")
        
        pt_inf = data.get("validation", {}).get("inference_results", {}).get("pytorch", {})
        onnx_inf = data.get("validation", {}).get("inference_results", {}).get("onnx", {})
        
        html = f"""<!DOCTYPE html><html><head><title>Training Validation Report - {self.exp_id}</title></head>
        <body style="font-family: Arial; margin: 40px;">
        <h1>Training Validation Report: {self.exp_id}</h1>
        <h2>Inference Comparativa (PT vs ONNX)</h2>
        <table border="1" cellpadding="10" style="border-collapse: collapse;">
            <tr>
                <th>Backend</th>
                <th>Detecciones</th>
                <th>Confidence Promedio</th>
                <th>Tiempo Inferencia (s)</th>
            </tr>
            <tr>
                <td>PyTorch (.pt)</td>
                <td>{pt_inf.get("total_detections", "N/A")}</td>
                <td>{pt_inf.get("avg_confidence", "N/A")}</td>
                <td>{pt_inf.get("avg_inference_time_s", "N/A")}</td>
            </tr>
            <tr>
                <td>ONNX Runtime (.onnx)</td>
                <td>{onnx_inf.get("total_detections", "N/A")}</td>
                <td>{onnx_inf.get("avg_confidence", "N/A")}</td>
                <td>{onnx_inf.get("avg_inference_time_s", "N/A")}</td>
            </tr>
        </table>
        
        <h2>Gráficas</h2>
        <p><i>Revisar el directorio validation/graphs/ para los PNG originales.</i></p>
        </body></html>
        """
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
