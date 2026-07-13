"""
src/synthetic_generation/engine.py
─────────────────────────────────────
Única responsabilidad: orquestar el motor completo de generación
sintética de extremo a extremo.

Flujo:
  1. Leer configuración YAML.
  2. Cargar dataset original (imágenes + etiquetas).
  3. Construir el pipeline de transformaciones desde el registro.
  4. Por cada imagen a generar:
       a. Seleccionar imagen fuente aleatoriamente.
       b. Crear semilla derivada de la semilla global.
       c. Ejecutar pipeline.
       d. Si inválida → reintentar (max_retries) → descartar.
       e. Exportar imagen, label, metadata.
       f. Generar preview si habilitado.
       g. Actualizar estadísticas.
  5. Guardar run_stats.json.
  6. Mostrar resumen en consola (rich).
"""

from __future__ import annotations
import os
import glob
import sys

import cv2
import numpy as np
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich import box

# Imports del motor
from metadata   import ImageMetadata
from statistics import RunStatistics
from registry   import TransformRegistry
from pipeline   import SyntheticPipeline
from exporter   import Exporter
from preview    import PreviewGenerator

console = Console(force_terminal=True, highlight=False)

VALID_IMG_EXT = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


# ── Carga de dataset ──────────────────────────────────────────────────────────

def _load_dataset(images_dir: str, labels_dir: str):
    """
    Carga las imágenes y polígonos YOLO Seg del dataset original.
    Devuelve lista de dicts con 'image_path', 'polygons', 'class_ids'.
    """
    image_paths = sorted([
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if p.endswith(VALID_IMG_EXT)
    ])

    records = []
    for img_path in image_paths:
        stem      = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path  = os.path.join(labels_dir, f"{stem}.txt")
        polygons  = []
        class_ids = []

        if os.path.isfile(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        coords = [float(v) for v in parts[1:]]
                        if len(coords) % 2 != 0 or len(coords) < 6:
                            continue
                        poly = list(zip(coords[0::2], coords[1::2]))
                        polygons.append(poly)
                        class_ids.append(cls_id)
                    except ValueError:
                        continue

        records.append({
            "image_path": img_path,
            "polygons":   polygons,
            "class_ids":  class_ids,
        })

    return records


# ── Motor principal ───────────────────────────────────────────────────────────

class SyntheticEngine:
    """Motor de generación sintética configurable desde YAML."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)

        self.base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )

    def _resolve(self, rel_path: str) -> str:
        return os.path.join(self.base_dir, rel_path)

    # ── Build del pipeline ────────────────────────────────────────────────────

    def _build_pipeline(self) -> SyntheticPipeline:
        transforms_cfg = self.cfg.get("transforms", {})
        transforms     = []

        for name, params in transforms_cfg.items():
            if not params.get("enabled", True):
                continue
            kwargs = {k: v for k, v in params.items() if k != "enabled"}
            t = TransformRegistry.build(name, **kwargs)
            transforms.append(t)
            console.print(f"   [cyan]+[/cyan] {name} (p={kwargs.get('probability', 1.0)})")

        return SyntheticPipeline(transforms)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, num_images: int | None = None) -> RunStatistics:
        eng_cfg  = self.cfg.get("engine",   {})
        pipe_cfg = self.cfg.get("pipeline", {})
        paths    = self.cfg.get("paths",    {})

        seed       = int(eng_cfg.get("seed",         42))
        n_images   = int(num_images or eng_cfg.get("num_images", 50))
        fmt        = str(eng_cfg.get("output_format", "jpg"))
        quality    = int(eng_cfg.get("output_quality", 95))
        max_retry  = int(eng_cfg.get("max_retries",    3))
        do_preview = bool(pipe_cfg.get("save_previews", True))
        prev_width = int(pipe_cfg.get("preview_max_width", 1280))

        console.print(f"\n[bold yellow]ASDGE — Adaptive Synthetic Dataset Generation Engine[/bold yellow]")
        console.print(f"  Config:  {self.config_path}")
        console.print(f"  Seed:    {seed}  |  Target: {n_images} imagenes\n")

        # Cargar dataset
        src_images = self._resolve(paths.get("source_images", "dataset/original"))
        src_labels = self._resolve(paths.get("source_labels", "dataset/labels"))
        dataset    = _load_dataset(src_images, src_labels)

        if not dataset:
            console.print("[bold red]ERROR: No se encontraron imágenes en el dataset.[/bold red]")
            sys.exit(1)

        console.print(f"  Dataset: {len(dataset)} imagenes originales cargadas.")

        # Construir pipeline
        console.print("\n[bold cyan]>> Construyendo pipeline de transformaciones...[/bold cyan]")
        pipeline = self._build_pipeline()

        # Inicializar módulos
        out_images   = self._resolve(paths.get("output_images",   "dataset/synthetic/images"))
        out_labels   = self._resolve(paths.get("output_labels",   "dataset/synthetic/labels"))
        out_previews = self._resolve(paths.get("output_previews", "dataset/synthetic/previews"))
        out_metadata = self._resolve(paths.get("output_metadata", "dataset/synthetic/metadata"))
        out_stats    = self._resolve(paths.get("run_stats",       "dataset/synthetic/run_stats.json"))

        exporter = Exporter(
            output_images_dir=out_images,
            output_labels_dir=out_labels,
            output_metadata_dir=out_metadata,
            output_format=fmt,
            output_quality=quality,
        )
        previewer = PreviewGenerator(out_previews, max_width=prev_width) if do_preview else None
        stats     = RunStatistics(config_path=self.config_path)
        stats.requested = n_images

        # RNG global
        global_rng = np.random.default_rng(seed)

        console.print(f"\n[bold cyan]>> Generando {n_images} imagenes sinteticas...[/bold cyan]")

        idx = 0
        with console.status("") as _:
            gen_count = 0
            for step in track(range(n_images), description="  Generando"):
                generated = False
                attempt   = 0

                while not generated and attempt <= max_retry:
                    attempt += 1
                    if attempt > 1:
                        stats.record_retry()

                    # Seleccionar fuente aleatoriamente
                    src = dataset[int(global_rng.integers(0, len(dataset)))]
                    img = cv2.imread(src["image_path"])
                    if img is None:
                        stats.record_error()
                        break

                    h, w = img.shape[:2]

                    # Semilla derivada (reproducible)
                    img_seed  = int(global_rng.integers(0, 2**32))
                    img_rng   = np.random.default_rng(img_seed)

                    # Ejecutar pipeline
                    img_out, polys_out, records, is_valid, issues = pipeline.run(
                        img, src["polygons"], img_rng
                    )

                    if not is_valid:
                        stats.record_discarded(reason="; ".join(issues))
                        continue

                    # Nombre del archivo de salida
                    gen_count += 1
                    filename = f"synthetic_{gen_count:05d}.{fmt}"

                    # Construir metadatos
                    meta = ImageMetadata(
                        source_image    = os.path.basename(src["image_path"]),
                        generated_image = filename,
                        seed            = img_seed,
                        width           = w,
                        height          = h,
                        n_polygons      = len(polys_out),
                    )
                    for rec in records:
                        meta.add_transform(rec)
                        if rec.applied:
                            stats.record_transform(rec.name)
                    meta.set_validation("valid", [])

                    # Exportar
                    exporter.export(
                        image     = img_out,
                        polygons  = polys_out,
                        class_ids = src["class_ids"],
                        filename  = filename,
                        meta      = meta,
                    )

                    # Preview
                    if previewer:
                        prev_name = f"preview_{gen_count:05d}.jpg"
                        previewer.generate(
                            original_image    = img,
                            synthetic_image   = img_out,
                            original_polygons = src["polygons"],
                            synthetic_polygons= polys_out,
                            transforms_applied= [r.to_dict() for r in records],
                            output_filename   = prev_name,
                        )

                    stats.record_generated()
                    generated = True

                if not generated and attempt > max_retry:
                    stats.record_discarded(reason="max_retries exceeded")

        stats.finish()
        stats.save(out_stats)
        self._print_summary(stats)
        return stats

    def _print_summary(self, stats: RunStatistics) -> None:
        d = stats.to_dict()
        dqi_color = "green" if d["success_rate_pct"] >= 95 else "yellow"

        table = Table(
            title="ASDGE RUN COMPLETED",
            box=box.DOUBLE_EDGE,
            border_style="magenta",
            show_header=False,
            padding=(0, 2),
        )
        table.add_column("Metrica", style="bold white", width=30)
        table.add_column("Valor",   style="bold",       width=20)

        table.add_row("Requested",        str(d["requested"]))
        table.add_row("Generated",        str(d["generated"]))
        table.add_row("Discarded",        str(d["discarded"]))
        table.add_row("Errors",           str(d["errors"]))
        table.add_row("Retries",          str(d["retries"]))
        table.add_row("Success Rate",     f"[{dqi_color}]{d['success_rate_pct']} %[/{dqi_color}]")
        table.add_row("Elapsed",          f"{d['elapsed_seconds']} s")

        transforms_str = ", ".join(
            f"{k}:{v}" for k, v in d["transforms_applied"].items()
        ) or "none"
        table.add_row("Transforms Used", transforms_str)

        console.print()
        console.print(table)
