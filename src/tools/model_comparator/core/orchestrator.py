import cv2
import os
from typing import List
from tools.model_comparator.core.report_model import ModelReport
from tools.model_comparator.analyzers.structural_inspector import StructuralInspector
from tools.model_comparator.adapters.factory import AdapterFactory
from tools.model_comparator.engine.benchmark_runner import BenchmarkRunner
from tools.model_comparator.renderers.visual_renderer import PolygonRenderer, BBoxRenderer, MosaicRenderer
from tools.model_comparator.reporters.json_csv_exporter import JSONExporter, CSVExporter
from tools.model_comparator.reporters.pdf_html_exporter import PDFHTMLExporter
from tools.model_comparator.analyzers.compatibility_analyzer import CompatibilityAnalyzer
import numpy as np

class Orchestrator:
    @staticmethod
    def run_comparison(model_paths: List[str], image_path: str, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        
        # Load image
        raw_image = cv2.imread(image_path)
        if raw_image is None:
            print(f"Error loading image {image_path}")
            return
            
        reports = []
        rendered_images = []
        
        for path in model_paths:
            print(f"\n[*] Processing model: {path}")
            
            # 1. Structural Analysis
            report = StructuralInspector.inspect(path)
            
            # 2. Adapter Factory
            adapter = AdapterFactory.get_adapter(report)
            
            # 3. Benchmark Engine & Inference
            post_data = BenchmarkRunner.run(report, adapter, raw_image, num_runs=5)
            
            # 4. Rendering
            if report.task == "segment":
                img_out = PolygonRenderer.render(raw_image, post_data, report.classes)
            else:
                img_out = BBoxRenderer.render(raw_image, post_data, report.classes)
                
            rendered_images.append(img_out)
            reports.append(report)
            
            # Optional: Compatibility analyzer against standard platform expected shape (1, 300, 6)
            compat_result = CompatibilityAnalyzer.analyze(report, {"output_shape": [1, 300, 6], "num_outputs": 1})
            with open(os.path.join(out_dir, f"compat_{report.model_name}.json"), "w") as f:
                import json
                json.dump(compat_result, f, indent=2)
                
        # Export Reports
        JSONExporter.export(reports, os.path.join(out_dir, "report.json"))
        CSVExporter.export(reports, os.path.join(out_dir, "report.csv"))
        try:
            PDFHTMLExporter.export_pdf(reports, os.path.join(out_dir, "report.pdf"))
        except Exception as e:
            print(f"Error exporting PDF: {e}")
        
        # Mosaic Rendering
        if len(rendered_images) >= 2:
            mosaic = MosaicRenderer.side_by_side(
                rendered_images[0], 
                rendered_images[1], 
                title_a=reports[0].model_name, 
                title_b=reports[1].model_name
            )
            cv2.imwrite(os.path.join(out_dir, "comparison.png"), mosaic)
            
        # Also save individual
        for r, img in zip(reports, rendered_images):
            cv2.imwrite(os.path.join(out_dir, f"render_{r.model_name}.png"), img)
            
        print(f"\n[+] Comparison completed. Results saved to {out_dir}")
