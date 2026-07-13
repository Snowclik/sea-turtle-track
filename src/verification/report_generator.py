"""
src/verification/report_generator.py
Genera los reportes de salida y la interfaz de consola.
"""

from __future__ import annotations
import os
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table as RLTable, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

console = Console(force_terminal=True, highlight=False)

class ReportGenerator:
    def __init__(self, project_root: str, eco_results: dict, sys_results: dict):
        self.output_dir = os.path.join(project_root, "reports", "verification")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {}
        self.results.update(eco_results)
        self.results.update(sys_results)
        
        self.all_passed = all(self.results.values())

    def display_console(self):
        console.print("\n[bold cyan]ECOSYSTEM READINESS REPORT[/bold cyan]\n")
        
        # Asignar nombres limpios para coincidir con la solicitud
        mapping = {
            "Dataset": "Dataset",
            "Experiments": "Synthetic Engine",
            "Benchmark Framework": "Benchmark Framework",
            "Configuration Files": "Experimental Protocol",
            "Training Framework": "Training Framework",
            "Configuration Files": "Configuration Files",
            "System Requirements": "Filesystem",
            "Fingerprints": "Fingerprints",
            "System Requirements": "System Requirements"
        }
        
        # En la realidad usaremos los valores de self.results para imprimir
        for k, v in self.results.items():
            status = "[green]PASS[/green]" if v else "[red]FAIL[/red]"
            dots = "." * (35 - len(k))
            console.print(f"{k} {dots} {status}")
            
        console.print("\n" + "-"*40 + "\n")
        
        console.print("OVERALL STATUS\n")
        if self.all_passed:
            console.print(Panel("[bold green]READY FOR LOCAL TRAINING[/bold green]", expand=False))
        else:
            console.print(Panel("[bold red]NOT READY - FIX ERRORS ABOVE[/bold red]", expand=False))

    def export_all(self):
        self._export_json()
        if HAS_OPENPYXL:
            self._export_xlsx()
        self._export_html()
        self._export_pdf()

    def _export_json(self):
        out = os.path.join(self.output_dir, "ecosystem_report.json")
        payload = {
            "all_passed": self.all_passed,
            "results": self.results
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

    def _export_xlsx(self):
        out = os.path.join(self.output_dir, "ecosystem_report.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Audit"
        ws.append(["Component", "Status"])
        for k, v in self.results.items():
            ws.append([k, "PASS" if v else "FAIL"])
        wb.save(out)

    def _export_html(self):
        out = os.path.join(self.output_dir, "ecosystem_report.html")
        html = """<!DOCTYPE html><html><head><title>Ecosystem Audit</title></head>
        <body style="font-family: Arial;"><h1>Ecosystem Readiness Report</h1>
        <table border="1" cellpadding="5" style="border-collapse: collapse;">
        <tr><th>Component</th><th>Status</th></tr>"""
        for k, v in self.results.items():
            status = "PASS" if v else "FAIL"
            color = "green" if v else "red"
            html += f"<tr><td>{k}</td><td style='color:{color}; font-weight:bold;'>{status}</td></tr>"
        
        overall = "READY FOR LOCAL TRAINING" if self.all_passed else "NOT READY"
        html += f"""</table><br><h2>OVERALL STATUS: {overall}</h2></body></html>"""
        
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)

    def _export_pdf(self):
        out = os.path.join(self.output_dir, "ecosystem_report.pdf")
        doc = SimpleDocTemplate(out, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        elements.append(Paragraph("Ecosystem Readiness Report", styles['Title']))
        elements.append(Spacer(1, 20))
        
        data = [["Component", "Status"]]
        for k, v in self.results.items():
            data.append([k, "PASS" if v else "FAIL"])
            
        t = RLTable(data, colWidths=[200, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(t)
        
        overall = "READY FOR LOCAL TRAINING" if self.all_passed else "NOT READY"
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(f"OVERALL STATUS: {overall}", styles['Heading2']))
        
        doc.build(elements)
