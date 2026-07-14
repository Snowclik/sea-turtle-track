"""
src/benchmark/campaign_report.py
Construye la comparativa final, genera los reportes y elige el OFFICIAL_MODEL.
"""

from __future__ import annotations
import os
import json
import shutil
from rich.console import Console

console = Console(highlight=False)

class CampaignReportGenerator:
    def __init__(self, campaign_id: str, project_root: str):
        self.campaign_id = campaign_id
        self.project_root = project_root
        self.campaign_dir = os.path.join(project_root, "campaigns", f"Campaign_{campaign_id}")
        self.leaderboard_path = os.path.join(project_root, "benchmark_results", "benchmark_leaderboard.json")

    def generate_and_crown(self):
        console.print(f"\n[bold magenta]Generando comparativa final y eligiendo OFFICIAL MODEL...[/bold magenta]")
        
        if not os.path.isfile(self.leaderboard_path):
            console.print("[red]✖ No existe el leaderboard, imposible comparar.[/red]")
            return
            
        with open(self.leaderboard_path, "r", encoding="utf-8") as f:
            leaderboard = json.load(f).get("experiments", {})
            
        # Filtrar solo los validados o entrenados de nuestra lista
        valid_datasets = ["baseline", "geometric_only", "photometric_only", "mixed"]
        
        # Debemos hacer matching. Recordar que los keys son los exp_id físicos (exp_2026...)
        # pero metadata.config_name tiene el nombre (ej: baseline.yaml)
        candidates = []
        for k, v in leaderboard.items():
            conf = v.get("metadata", {}).get("config_name", "").replace(".yaml", "")
            if conf in valid_datasets and v.get("metadata", {}).get("status") in ["TRAINED", "VALIDATED"]:
                candidates.append({
                    "id": k,
                    "dataset": conf,
                    "status": v["metadata"]["status"],
                    "mAP50": v.get("training_metrics", {}).get("mAP_50") or 0.0,
                    "mAP50_95": v.get("training_metrics", {}).get("mAP_50_95") or 0.0,
                    "precision": v.get("training_metrics", {}).get("precision") or 0.0,
                    "recall": v.get("training_metrics", {}).get("recall") or 0.0,
                    "f1_score": v.get("training_metrics", {}).get("f1_score") or 0.0,
                    "inference_s": 999.0, # fallback
                    "training_s": v.get("training_metrics", {}).get("training_time_s") or 0.0,
                    "size_mb": v.get("training_metrics", {}).get("model_size_mb") or 0.0,
                    "params_m": v.get("training_metrics", {}).get("params_m") or 0.0
                })
                
        # Completar inference_s desde los reportes de validación
        for c in candidates:
            fp_path = os.path.join(self.project_root, "experiments", c["id"], "validation", "training_validation_fingerprint.json")
            if os.path.isfile(fp_path):
                with open(fp_path, "r", encoding="utf-8") as f:
                    fp = json.load(f)
                    inf = fp.get("inference_results", {}).get("onnx", {}).get("avg_inference_time_s")
                    if inf is not None:
                        c["inference_s"] = inf
                        
        # 1. ORDENAR (Mayor mAP50-95, Mayor F1, Mayor Precision, Menor tiempo de inferencia)
        candidates.sort(key=lambda x: (
            x["mAP50_95"],
            x["f1_score"],
            x["precision"],
            -x["inference_s"]
        ), reverse=True)
        
        # El campeón es el índice 0
        if not candidates:
            console.print("[red]✖ No hay candidatos válidos para competir.[/red]")
            return
            
        champion = candidates[0]
        console.print(f"[bold yellow]🏆 OFFICIAL MODEL CROWNED: {champion['dataset']} (mAP50-95: {champion['mAP50_95']})[/bold yellow]")
        
        # Marcar y copiar el oficial
        self._set_official(champion)
        
        # Escribir json
        out_json = os.path.join(self.campaign_dir, "campaign_summary.json")
        payload = {
            "campaign": f"Campaign_{self.campaign_id}",
            "official_model_winner": champion["dataset"],
            "official_model_id": champion["id"],
            "ranking": candidates
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
            
        # Generar HTML
        self._export_html(payload)
        
        console.print(f"  [green]✔ Reportes de campaña escritos en {self.campaign_dir}[/green]")
        
    def _set_official(self, champion: dict):
        # Crear la carpeta final OFFICIAL_MODEL
        off_dir = os.path.join(self.project_root, "OFFICIAL_MODEL")
        os.makedirs(off_dir, exist_ok=True)
        
        exp_dir = os.path.join(self.project_root, "experiments", champion["id"])
        best_onnx = os.path.join(exp_dir, "training_results", "exported_models", "best.onnx")
        best_pt = os.path.join(exp_dir, "training_results", "exported_models", "best.pt")
        
        if os.path.isfile(best_onnx):
            shutil.copy(best_onnx, os.path.join(off_dir, "yolo26n-seg_asgf_official.onnx"))
        if os.path.isfile(best_pt):
            shutil.copy(best_pt, os.path.join(off_dir, "yolo26n-seg_asgf_official.pt"))
            
        with open(os.path.join(off_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(champion, f, indent=4)
            
    def _export_html(self, payload: dict):
        out = os.path.join(self.campaign_dir, "campaign_report.html")
        html = f"""<!DOCTYPE html><html><head><title>{payload['campaign']} - Scientific Evaluation</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #f9fafb; color: #111827; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            h1, h2 {{ color: #1f2937; }}
            .champion {{ border: 2px solid #fbbf24; background-color: #fffbeb; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
            th {{ background-color: #f3f4f6; font-weight: 600; }}
            tr:hover {{ background-color: #f9fafb; }}
            .rank-1 {{ font-weight: bold; color: #047857; }}
        </style>
        </head>
        <body>
            <div class="card champion">
                <h1>🏆 Official Model: {payload['official_model_winner']}</h1>
                <p>El mejor modelo ha sido exportado al directorio <code>OFFICIAL_MODEL/</code> para despliegue en React.</p>
            </div>
            
            <div class="card">
                <h2>Tabla Comparativa Científica (Ranking Final)</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Dataset</th>
                            <th>mAP50-95</th>
                            <th>mAP50</th>
                            <th>F1 Score</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>Inf. Time (s)</th>
                            <th>Train Time (s)</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        for idx, c in enumerate(payload["ranking"]):
            rank_class = "rank-1" if idx == 0 else ""
            html += f"""
                        <tr class="{rank_class}">
                            <td>#{idx+1}</td>
                            <td>{c['dataset']}</td>
                            <td>{c['mAP50_95']}</td>
                            <td>{c['mAP50']}</td>
                            <td>{c['f1_score']}</td>
                            <td>{c['precision']}</td>
                            <td>{c['recall']}</td>
                            <td>{c['inference_s']}</td>
                            <td>{c['training_s']}</td>
                            <td>{c['status']}</td>
                        </tr>
            """
        
        html += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
