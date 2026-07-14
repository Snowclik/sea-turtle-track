import json
import dataclasses
from typing import List
from tools.model_comparator.core.report_model import ModelReport

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

class JSONExporter:
    @staticmethod
    def export(reports: List[ModelReport], out_path: str):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(reports, f, cls=EnhancedJSONEncoder, indent=2)
            
class CSVExporter:
    @staticmethod
    def export(reports: List[ModelReport], out_path: str):
        import csv
        if not reports:
            return
        
        with open(out_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Model", "Task", "Size_MB", "Nodes", "FPS", 
                "Avg_Latency_ms", "RAM_MB", "CPU_pct", "Detections", "Polygons"
            ])
            for r in reports:
                writer.writerow([
                    r.model_name,
                    r.task,
                    f"{r.size_mb:.2f}",
                    r.num_nodes,
                    f"{r.performance.fps:.2f}",
                    f"{r.performance.avg_latency_ms:.2f}",
                    f"{r.performance.ram_usage_mb:.2f}",
                    f"{r.performance.cpu_usage_percent:.2f}",
                    r.results.num_detections,
                    r.results.num_polygons
                ])
