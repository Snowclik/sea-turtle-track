"""
src/training/metrics_collector.py
Recolecta y formatea las métricas del entrenamiento para el Benchmark.
(Esqueleto).
"""

from __future__ import annotations

class MetricsCollector:
    """Prepara la estructura de resultados compatible con el Leaderboard."""
    
    def __init__(self):
        self.metrics = {
            "mAP_50": None,
            "mAP_50_95": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "training_time_s": None,
            "inference_time_ms": None,
            "model_size_mb": None,
            "params_m": None
        }
        
    def collect(self, yolo_run_dir: str, total_time: float, model_instance) -> dict:
        """Extrae las métricas del archivo results.csv generado por YOLO."""
        import os
        import csv
        
        self.metrics["training_time_s"] = round(total_time, 2)
        
        results_csv = os.path.join(yolo_run_dir, "results.csv")
        if os.path.isfile(results_csv):
            with open(results_csv, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = [h.strip().lower() for h in next(reader)]
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    
                    # Funciones de búsqueda
                    def get_val(keywords):
                        for i, h in enumerate(headers):
                            if any(k in h for k in keywords):
                                return float(last_row[i])
                        return None
                        
                    p = get_val(["metrics/precision"])
                    r = get_val(["metrics/recall"])
                    map50 = get_val(["map50(m)", "map50"])
                    map95 = get_val(["map50-95(m)", "map50-95"])
                    
                    if p is not None: self.metrics["precision"] = round(p, 4)
                    if r is not None: self.metrics["recall"] = round(r, 4)
                    if map50 is not None: self.metrics["mAP_50"] = round(map50, 4)
                    if map95 is not None: self.metrics["mAP_50_95"] = round(map95, 4)
                    
                    if p and r and (p + r) > 0:
                        self.metrics["f1_score"] = round(2 * (p * r) / (p + r), 4)

        # Extraer Params
        if hasattr(model_instance, 'info'):
            try:
                total_params = sum(p.numel() for p in model_instance.model.parameters())
                self.metrics["params_m"] = round(total_params / 1e6, 2)
                self.metrics["model_size_mb"] = round((total_params * 4) / (1024**2), 2)
            except:
                pass

        return self.metrics
