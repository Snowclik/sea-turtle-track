"""
src/synthetic_generation/verifier/verifier.py
──────────────────────────────────────────────
Orquestador principal de la validación post-generación.
"""

from __future__ import annotations
import os
import glob
import json
import yaml
import cv2
import time

from rich.console import Console
from rich.table import Table
from rich import box
from rich.progress import track

from .fingerprint import PipelineFingerprint
from .reporting import Reporter
from .visualizer import Visualizer

console = Console(force_terminal=True, highlight=False)
VALID_EXT = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

class ExperimentVerifier:
    def __init__(self, experiment_dir: str):
        self.experiment_dir = os.path.abspath(experiment_dir)
        self.config_path = os.path.join(self.experiment_dir, "config.yaml")
        
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
            
        self.val_cfg = self.cfg.get("validation", {})
        self.min_success = float(self.val_cfg.get("min_success_rate", 99.0))
        self.max_errors = int(self.val_cfg.get("max_errors", 0))

    def _validate_image(self, img_path: str, lbl_path: str, meta_path: str) -> list[str]:
        """Audita una única imagen sintética. Devuelve lista de errores."""
        errors = []
        
        # 1. Integridad de archivo
        img = cv2.imread(img_path)
        if img is None:
            return ["Archivo de imagen corrupto o ilegible."]

        # 2. Etiqueta
        if not os.path.isfile(lbl_path):
            errors.append("Falta archivo de etiqueta YOLO.")
        
        # 3. Metadatos y Coherencia
        if not os.path.isfile(meta_path):
            errors.append("Falta archivo de metadatos JSON.")
        else:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                errors.append("Metadatos JSON corruptos.")
                meta = {}
        
        # 4. Validar Polígonos de la etiqueta
        polygons = []
        if os.path.isfile(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        coords = [float(v) for v in parts[1:]]
                        poly = list(zip(coords[0::2], coords[1::2]))
                        polygons.append(poly)
                        
                        # Validaciones geométricas
                        if len(poly) < 3:
                            errors.append(f"Polígono degenerado (<3 vértices).")
                        
                        # Área shoelace
                        area = 0.0
                        n = len(poly)
                        for i in range(n):
                            x1, y1 = poly[i]
                            x2, y2 = poly[(i+1)%n]
                            area += x1*y2 - x2*y1
                        area = abs(area)/2.0
                        if area <= 0.0:
                            errors.append(f"Área del polígono es 0.")
                            
                        # Límites
                        for x, y in poly:
                            if not (0.0 <= x <= 1.0) or not (0.0 <= y <= 1.0):
                                errors.append(f"Vértice fuera de [0, 1]: ({x},{y}).")
        
        # Coherencia de conteo original vs sintético (si meta fue cargado y no faltan labels)
        if meta and "n_polygons" in meta:
            if len(polygons) != meta["n_polygons"]:
                errors.append(f"Mismatch de objetos: metadata dice {meta['n_polygons']}, label tiene {len(polygons)}.")

        return errors

    def run(self) -> dict:
        console.print(f"\n[bold magenta]Auditoría del ASDGE — {os.path.basename(self.experiment_dir)}[/bold magenta]")
        
        start_time = time.time()
        
        images_dir = os.path.join(self.experiment_dir, "images")
        labels_dir = os.path.join(self.experiment_dir, "labels")
        meta_dir = os.path.join(self.experiment_dir, "metadata")
        
        image_paths = sorted([
            p for p in glob.glob(os.path.join(images_dir, "*"))
            if p.endswith(VALID_EXT)
        ])
        
        total = len(image_paths)
        valid_count = 0
        corrupted_files = 0
        missing_labels = 0
        invalid_polygons = 0
        metadata_mismatches = 0
        critical_errors = []
        
        console.print(f"  Validando {total} imágenes sintéticas...")
        
        for img_path in track(image_paths, description="  Auditoría"):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(labels_dir, f"{stem}.txt")
            meta_path = os.path.join(meta_dir, f"{stem}.json")
            
            errors = self._validate_image(img_path, lbl_path, meta_path)
            
            if not errors:
                valid_count += 1
            else:
                for err in errors:
                    critical_errors.append(f"{stem}: {err}")
                    if "corrupto" in err:
                        corrupted_files += 1
                    elif "Falta archivo de etiqueta" in err:
                        missing_labels += 1
                    elif "Mismatch" in err:
                        metadata_mismatches += 1
                    else:
                        invalid_polygons += 1

        success_rate = (valid_count / total * 100.0) if total > 0 else 0.0
        passed = (success_rate >= self.min_success) and (len(critical_errors) <= self.max_errors)

        # Generar Pipeline Fingerprint
        console.print("  [cyan]>> Generando Pipeline Fingerprint...[/cyan]")
        fp_gen = PipelineFingerprint(self.experiment_dir)
        fingerprint_data = fp_gen.generate()
        
        # Generar Visual Samples
        num_samples = int(self.val_cfg.get("visual_samples", 10))
        if num_samples > 0:
            console.print("  [cyan]>> Seleccionando muestras visuales...[/cyan]")
            vis = Visualizer(self.experiment_dir, num_samples)
            vis.generate_samples()
            
        results = {
            "experiment_id": fingerprint_data.get("experiment_id", ""),
            "total_generated": total,
            "valid_images": valid_count,
            "corrupted_files": corrupted_files,
            "missing_labels": missing_labels,
            "invalid_polygons": invalid_polygons,
            "metadata_mismatches": metadata_mismatches,
            "critical_errors": critical_errors,
            "success_rate_pct": round(success_rate, 2),
            "passed": passed,
            "pipeline_fingerprint": fingerprint_data.get("pipeline_fingerprint", ""),
            "elapsed_seconds": round(time.time() - start_time, 2)
        }
        
        # Guardar summary
        val_dir = os.path.join(self.experiment_dir, "validation")
        with open(os.path.join(val_dir, "validation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
        # Generar reportes
        if self.val_cfg.get("generate_reports", True):
            console.print("  [cyan]>> Generando reportes PDF y HTML...[/cyan]")
            reporter = Reporter(self.experiment_dir)
            reporter.generate_html(results)
            reporter.generate_pdf(results)
            
        self._print_summary(results)
        return results

    def _print_summary(self, results: dict):
        ok_color = "green" if results["passed"] else "red"
        
        table = Table(
            title="ASDGE VALIDATION RESULTS",
            box=box.DOUBLE_EDGE,
            border_style="cyan",
            show_header=False,
            padding=(0, 2),
        )
        table.add_column("Metrica", style="bold white", width=32)
        table.add_column("Valor",   style="bold",       width=22)

        table.add_row("Experiment ID", results["experiment_id"])
        table.add_row("Total Images", str(results["total_generated"]))
        table.add_row("Valid Images", str(results["valid_images"]))
        table.add_row("Validation Errors", str(len(results["critical_errors"])))
        table.add_row("Success Rate", f"[{ok_color}]{results['success_rate_pct']} %[/{ok_color}]")
        table.add_row("Status", f"[{ok_color}]{'PASSED' if results['passed'] else 'FAILED'}[/{ok_color}]")
        table.add_row("Fingerprint", results["pipeline_fingerprint"][:16] + "...")

        console.print()
        console.print(table)
        
        if not results["passed"]:
            console.print("\n[bold red]ERRORES CRÍTICOS ENCONTRADOS:[/bold red]")
            for err in results["critical_errors"][:10]:
                console.print(f"  - {err}")
            if len(results["critical_errors"]) > 10:
                console.print(f"  ... y {len(results['critical_errors']) - 10} más.")
