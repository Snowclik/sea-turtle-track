from fpdf import FPDF
from typing import List
from tools.model_comparator.core.report_model import ModelReport

class PDFHTMLExporter:
    @staticmethod
    def export_pdf(reports: List[ModelReport], out_path: str):
        if not reports:
            return
            
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "Universal Model Comparator - Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(10)
        
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "Scientific Comparison Table", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 10)
        
        # Table Header
        headers = ["Property"] + [r.model_name[:15] for r in reports]
        
        def add_row(prop: str, values: List[str]):
            pdf.set_font("helvetica", "B", 10)
            pdf.cell(40, 10, prop, border=1)
            pdf.set_font("helvetica", "", 10)
            col_width = (pdf.epw - 40) / len(values)
            for v in values:
                pdf.cell(col_width, 10, str(v)[:20], border=1)
            pdf.ln(10)

        # Build rows
        add_row("Task", [r.task for r in reports])
        add_row("Size (MB)", [f"{r.size_mb:.2f}" for r in reports])
        add_row("Nodes", [str(r.num_nodes) for r in reports])
        add_row("Opset", [str(r.onnx_opset) for r in reports])
        add_row("FPS", [f"{r.performance.fps:.2f}" for r in reports])
        add_row("Avg Latency (ms)", [f"{r.performance.avg_latency_ms:.2f}" for r in reports])
        add_row("RAM Usage (MB)", [f"{r.performance.ram_usage_mb:.2f}" for r in reports])
        add_row("CPU %", [f"{r.performance.cpu_usage_percent:.2f}" for r in reports])
        add_row("Detections", [str(r.results.num_detections) for r in reports])
        add_row("Polygons", [str(r.results.num_polygons) for r in reports])
        
        pdf.output(out_path)
