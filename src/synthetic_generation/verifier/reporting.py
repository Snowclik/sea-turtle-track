"""
src/synthetic_generation/verifier/reporting.py
───────────────────────────────────────────────
Genera los reportes HTML y PDF con los resultados de la validación.
"""

from __future__ import annotations
import os
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table as RLTable, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


class Reporter:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.out_dir = os.path.join(experiment_dir, "validation")
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_html(self, results: Dict[str, Any]) -> str:
        out_path = os.path.join(self.out_dir, "validation_report.html")
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ASDGE Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
        h1, h2 {{ color: #2c3e50; }}
        .success {{ color: #27ae60; font-weight: bold; }}
        .error {{ color: #c0392b; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 60%; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>ASDGE Validation Report</h1>
    <p>Experiment ID: <b>{results.get('experiment_id', 'N/A')}</b></p>
    
    <h2>Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Images Generated</td><td>{results.get('total_generated', 0)}</td></tr>
        <tr><td>Valid Images</td><td>{results.get('valid_images', 0)}</td></tr>
        <tr><td>Corrupted Files</td><td>{results.get('corrupted_files', 0)}</td></tr>
        <tr><td>Missing Labels</td><td>{results.get('missing_labels', 0)}</td></tr>
        <tr><td>Invalid Polygons</td><td>{results.get('invalid_polygons', 0)}</td></tr>
        <tr><td>Metadata Mismatches</td><td>{results.get('metadata_mismatches', 0)}</td></tr>
        <tr><td>Validation Success Rate</td><td class="{'success' if results.get('passed', False) else 'error'}">{results.get('success_rate_pct', 0.0)}%</td></tr>
        <tr><td>Passed</td><td class="{'success' if results.get('passed', False) else 'error'}">{results.get('passed', False)}</td></tr>
    </table>
    
    <h2>Errors</h2>
    <ul>
        {"".join(f"<li>{e}</li>" for e in results.get("critical_errors", [])) or "<li>No critical errors.</li>"}
    </ul>
</body>
</html>
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html)
        return out_path

    def generate_pdf(self, results: Dict[str, Any]) -> str:
        out_path = os.path.join(self.out_dir, "validation_report.pdf")
        doc = SimpleDocTemplate(out_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("ASDGE Validation Report", styles['Title']))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph(f"Experiment ID: {results.get('experiment_id', 'N/A')}", styles['Normal']))
        elements.append(Spacer(1, 20))

        data = [
            ["Metric", "Value"],
            ["Total Images", str(results.get('total_generated', 0))],
            ["Valid Images", str(results.get('valid_images', 0))],
            ["Corrupted Files", str(results.get('corrupted_files', 0))],
            ["Missing Labels", str(results.get('missing_labels', 0))],
            ["Invalid Polygons", str(results.get('invalid_polygons', 0))],
            ["Metadata Mismatches", str(results.get('metadata_mismatches', 0))],
            ["Success Rate", f"{results.get('success_rate_pct', 0.0)}%"],
            ["Passed", str(results.get('passed', False))]
        ]

        t = RLTable(data, colWidths=[200, 150])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)

        doc.build(elements)
        return out_path
