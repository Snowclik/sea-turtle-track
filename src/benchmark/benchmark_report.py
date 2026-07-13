"""
src/benchmark/benchmark_report.py
Generador de HTML, PDF, Excel y JSON para el Leaderboard.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table as RLTable, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from .comparator import BenchmarkComparator


class BenchmarkReporter:
    def __init__(self, leaderboard_data: Dict[str, Any], output_dir: str):
        self.data = leaderboard_data
        self.output_dir = output_dir
        self.comparator = BenchmarkComparator(leaderboard_data)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_json(self):
        out_path = os.path.join(self.output_dir, "benchmark_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
        return out_path

    def generate_excel(self):
        if not HAS_OPENPYXL:
            return None
            
        out_path = os.path.join(self.output_dir, "benchmark_summary.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Leaderboard"
        
        headers = ["Exp ID", "Date", "Config", "Status", "Images", "mAP@50", "mAP@50-95", "Precision", "Recall"]
        ws.append(headers)
        
        ranked = self.comparator.get_ranked_experiments(metric="date")
        for exp_id, record in ranked:
            meta = record.get("metadata", {})
            stats = record.get("dataset_stats", {})
            train = record.get("training_metrics", {})
            
            row = [
                exp_id,
                meta.get("date", ""),
                meta.get("config_name", ""),
                meta.get("status", ""),
                stats.get("num_images", 0),
                train.get("mAP_50", ""),
                train.get("mAP_50_95", ""),
                train.get("precision", ""),
                train.get("recall", "")
            ]
            ws.append(row)
            
        wb.save(out_path)
        return out_path

    def generate_html(self):
        out_path = os.path.join(self.output_dir, "benchmark_report.html")
        ranked = self.comparator.get_ranked_experiments(metric="date")
        
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Leaderboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
        h1, h2 { color: #2c3e50; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Benchmark Leaderboard</h1>
    <table>
        <tr>
            <th>Exp ID</th><th>Date</th><th>Config</th><th>Images</th><th>Status</th><th>mAP@50</th>
        </tr>
"""
        for exp_id, record in ranked:
            meta = record.get("metadata", {})
            stats = record.get("dataset_stats", {})
            train = record.get("training_metrics", {})
            
            html += f"""
        <tr>
            <td>{exp_id}</td>
            <td>{meta.get("date", "")}</td>
            <td>{meta.get("config_name", "")}</td>
            <td>{stats.get("num_images", 0)}</td>
            <td>{meta.get("status", "")}</td>
            <td>{train.get("mAP_50", "N/A")}</td>
        </tr>
"""
        html += """
    </table>
</body>
</html>
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return out_path

    def generate_pdf(self):
        out_path = os.path.join(self.output_dir, "benchmark_report.pdf")
        doc = SimpleDocTemplate(out_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Benchmark Leaderboard", styles['Title']))
        elements.append(Spacer(1, 20))

        data = [["Exp ID", "Config", "Images", "Status", "mAP@50"]]
        
        ranked = self.comparator.get_ranked_experiments(metric="date")
        for exp_id, record in ranked:
            meta = record.get("metadata", {})
            stats = record.get("dataset_stats", {})
            train = record.get("training_metrics", {})
            
            data.append([
                exp_id[:12]+"...", 
                meta.get("config_name", ""),
                str(stats.get("num_images", 0)),
                meta.get("status", ""),
                str(train.get("mAP_50", "N/A"))
            ])

        t = RLTable(data, colWidths=[120, 100, 70, 100, 80])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)

        doc.build(elements)
        return out_path
