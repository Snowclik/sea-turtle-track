"""
src/synthetic_generation/engine.py
────────────────────────────────────
Única responsabilidad: orquestar el motor completo de generación
sintética de extremo a extremo.

Flujo:
  1. Leer configuración YAML.
  2. Crear experimento (ID único + carpeta).
  3. Inicializar SeedManager con la semilla global.
  4. Cargar dataset original (imágenes + polígonos de segmentación).
  5. Construir pipeline desde el registro de transformaciones.
  6. Por cada imagen a generar:
       a. Seleccionar fuente (SeedManager).
       b. Obtener semilla derivada (SeedManager).
       c. Ejecutar pipeline.
       d. Si inválida → reintentar (max_retries) → descartar.
       e. Exportar imagen, label YOLO Seg, metadata JSON.
       f. Generar preview si habilitado.
       g. Actualizar estadísticas.
  7. Guardar run_stats.json + experiment_summary.json.
  8. Mostrar resumen en consola (rich).
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
from rich.panel import Panel
from rich import box

from metadata     import ImageMetadata
from statistics   import RunStatistics
from seed_manager import SeedManager
from experiment   import Experiment
from registry     import TransformRegistry
from pipeline     import SyntheticPipeline
from exporter     import Exporter
from preview      import PreviewGenerator

console   = Console(force_terminal=True, highlight=False)
VALID_EXT = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


# ── Carga de dataset ──────────────────────────────────────────────────────────

def _load_dataset(images_dir: str, labels_dir: str) -> list:
    """
    Carga imágenes + polígonos YOLO Seg del dataset fuente.
    Devuelve: [{"image_path", "polygons", "class_ids"}, ...]
    """
    image_paths = sorted([
        p for p in glob.glob(os.path.join(images_dir, "*"))
        if p.endswith(VALID_EXT)
    ])
    records = []
    for img_path in image_paths:
        stem      = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path  = os.path.join(labels_dir, f"{stem}.txt")
        polygons, class_ids = [], []

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
                        polygons.append(list(zip(coords[0::2], coords[1::2])))
                        class_ids.append(cls_id)
                    except ValueError:
                        continue

        records.append({
            "image_path": img_path,
            "polygons":   polygons,
            "class_ids":  class_ids,
        })
    return records


# ── Motor ─────────────────────────────────────────────────────────────────────

class SyntheticEngine:
    """
    Motor de generación sintética configurable desde YAML.
    Sin estado mutable entre runs: cada llamada a run() es independiente.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = yaml.safe_load(f)
        self._base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )

    def _resolve(self, rel: str) -> str:
        return os.path.join(self._base_dir, rel)

    # ── Build pipeline ────────────────────────────────────────────────────────

    def _build_pipeline(self) -> SyntheticPipeline:
        transforms_cfg = self.cfg.get("transforms", {})
        transforms     = []
        console.print("\n[bold cyan]>> Construyendo pipeline de transformaciones...[/bold cyan]")
        for name, params in transforms_cfg.items():
            if not params.get("enabled", True):
                console.print(f"   [dim]- {name} (deshabilitado)[/dim]")
                continue
            kwargs = {k: v for k, v in params.items() if k != "enabled"}
            t = TransformRegistry.build(name, **kwargs)
            transforms.append(t)
            console.print(f"   [cyan]+[/cyan] {name} [{t.category}] (p={kwargs.get('probability', 1.0)})")
        return SyntheticPipeline(transforms)

    # ── API Adaptativa (Stateful) ─────────────────────────────────────────────

    def setup(self, experiment_id: str | None = None) -> None:
        """Inicializa el estado del motor para generación incremental."""
        if hasattr(self, 'exp') and self.exp is not None:
            return  # Ya inicializado
            
        eng_cfg  = self.cfg.get("engine",   {})
        pipe_cfg = self.cfg.get("pipeline", {})
        paths    = self.cfg.get("paths",    {})

        self.global_seed = int(eng_cfg.get("seed", 42))
        self.fmt         = str(eng_cfg.get("output_format",  "jpg"))
        self.quality     = int(eng_cfg.get("output_quality", 95))
        self.max_retry   = int(eng_cfg.get("max_retries",    3))
        self.do_preview  = bool(pipe_cfg.get("save_previews",      True))
        self.prev_width  = int(pipe_cfg.get("preview_max_width",   1280))

        # ── Experimento ───────────────────────────────────────────────────────
        exp_base = self._resolve("experiments")
        
        # Le permitimos forzar un ID para resumir, si no el constructor genera uno
        if experiment_id:
            # Reconstruimos la misma logica de 'experiment.py' pero forzando el ID
            self.exp = Experiment(base_experiments_dir=exp_base, config_path=self.config_path)
            self.exp.experiment_id = experiment_id
            self.exp.experiment_dir = os.path.join(exp_base, experiment_id)
            self.exp.images_dir = os.path.join(self.exp.experiment_dir, "images")
            self.exp.labels_dir = os.path.join(self.exp.experiment_dir, "labels")
            self.exp.metadata_dir = os.path.join(self.exp.experiment_dir, "metadata")
            self.exp.previews_dir = os.path.join(self.exp.experiment_dir, "previews")
            self.exp.stats_dir = os.path.join(self.exp.experiment_dir, "stats")
            self.exp.summary_path = os.path.join(self.exp.experiment_dir, "experiment_summary.json")
            # Asegurar que existan
            os.makedirs(self.exp.experiment_dir, exist_ok=True)
            for d in [self.exp.images_dir, self.exp.labels_dir, self.exp.metadata_dir, self.exp.previews_dir, self.exp.stats_dir]:
                os.makedirs(d, exist_ok=True)
        else:
            self.exp = Experiment(base_experiments_dir=exp_base, config_path=self.config_path)

        console.print(f"\n[bold magenta]ASDGE — Adaptive Synthetic Dataset Generation Engine[/bold magenta]")
        console.print(f"  Experiment:  [bold]{self.exp.experiment_id}[/bold] (Stateful Mode)")
        console.print(f"  Config:      {self.config_path}")
        console.print(f"  Seed:        {self.global_seed}")
        console.print(f"  Output:      {self.exp.experiment_dir}")

        # ── Dataset ───────────────────────────────────────────────────────────
        # Forzar origen del dataset a splits/train para aislamiento riguroso
        base_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        src_images = os.path.join(base_dir_path, "dataset", "splits", "train", "images")
        src_labels = os.path.join(base_dir_path, "dataset", "splits", "train", "labels")
        
        # Fallback a original si no existe
        if not os.path.exists(src_images):
            src_images = self._resolve(paths.get("source_images", "dataset/original"))
            src_labels = self._resolve(paths.get("source_labels", "dataset/labels"))
            
        self.dataset = _load_dataset(src_images, src_labels)
        if not self.dataset:
            console.print("[bold red]ERROR: No se encontraron imágenes en el dataset.[/bold red]")
            sys.exit(1)
        console.print(f"  Dataset:     {len(self.dataset)} imagenes originales (Train) cargadas.")

        # ── Pipeline ──────────────────────────────────────────────────────────
        self.pipeline = self._build_pipeline()

        # ── Módulos de Estado ─────────────────────────────────────────────────
        self.seed_mgr = SeedManager(self.global_seed)
        self.exporter = Exporter(
            output_images_dir   = self.exp.images_dir,
            output_labels_dir   = self.exp.labels_dir,
            output_metadata_dir = self.exp.metadata_dir,
            output_format       = self.fmt,
            output_quality      = self.quality,
        )
        self.previewer = PreviewGenerator(self.exp.previews_dir, max_width=self.prev_width) if self.do_preview else None
        self.stats = RunStatistics(config_path=self.config_path)
        self.gen_count = 0
        self.histogram_history = []

    def _get_phash(self, img_out: np.ndarray) -> np.ndarray:
        """Calcula mapa estructural (geométrico) de muy baja resolución."""
        gray = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        norm = cv2.normalize(resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return norm

    def _is_diverse(self, img_out: np.ndarray) -> bool:
        """
        Filtro de diversidad ligero.
        Considera color (HSV) y estructura geométrica/escala (pHash).
        """
        if not hasattr(self, 'history_phash'):
            self.history_phash = []
            
        hsv = cv2.cvtColor(img_out, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        phash = self._get_phash(img_out)
        
        for prev_hist, prev_phash in zip(self.histogram_history, self.history_phash):
            dist_color = cv2.compareHist(hist, prev_hist, cv2.HISTCMP_BHATTACHARYYA)
            mse_struct = np.mean((phash - prev_phash) ** 2)
            
            # Si tanto el color como la estructura son casi idénticos, rechazar
            if dist_color < 0.05 and mse_struct < 0.02:
                return False
                
        self.histogram_history.append(hist)
        self.history_phash.append(phash)
        return True

    def step(self, batch_size: int) -> Experiment:
        """Genera un lote adicional de imágenes sintéticas manteniendo el estado anterior."""
        if not hasattr(self, 'exp') or self.exp is None:
            self.setup()
            
        self.stats.requested += batch_size
        console.print(f"\n[bold cyan]>> [Step] Generando {batch_size} imagenes sinteticas (Total acumulado esperado: {self.stats.requested})...[/bold cyan]")

        for _ in track(range(batch_size), description="  Generando"):
            generated = False
            attempt   = 0

            while not generated and attempt <= self.max_retry:
                attempt += 1
                if attempt > 1:
                    self.stats.record_retry()

                # Selección de fuente
                src_idx  = self.seed_mgr.select_source_index(len(self.dataset))
                src      = self.dataset[src_idx]
                img      = cv2.imread(src["image_path"])
                if img is None:
                    self.stats.record_error()
                    break

                h, w = img.shape[:2]

                # Semilla derivada por imagen
                img_seed = self.seed_mgr.next_image_seed()
                img_rng  = self.seed_mgr.image_rng(img_seed)

                # Pipeline
                img_out, polys_out, records, is_valid, issues = self.pipeline.run(
                    img, src["polygons"], img_rng
                )

                if not is_valid:
                    self.stats.record_discarded(reason="; ".join(issues))
                    continue
                    
                # Filtro de Diversidad (Rechazar imágenes redundantes)
                if not self._is_diverse(img_out):
                    self.stats.record_discarded(reason="redundant_image")
                    continue

                # Nombre del artefacto
                self.gen_count += 1
                filename   = f"synthetic_{self.gen_count:05d}.{self.fmt}"

                # Metadata
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
                        self.stats.record_transform(rec.name)
                meta.set_validation("valid", [])

                # Exportar
                self.exporter.export(
                    image     = img_out,
                    polygons  = polys_out,
                    class_ids = src["class_ids"],
                    filename  = filename,
                    meta      = meta,
                )

                # Preview
                if self.previewer:
                    self.previewer.generate(
                        original_image     = img,
                        synthetic_image    = img_out,
                        original_polygons  = src["polygons"],
                        synthetic_polygons = polys_out,
                        transforms_applied = [r.to_dict() for r in records],
                        output_filename    = f"preview_{self.gen_count:05d}.jpg",
                    )

                self.stats.record_generated()
                generated = True

            if not generated and attempt > self.max_retry:
                self.stats.record_discarded(reason="max_retries exceeded")
                
        # Siempre guardamos el estado tras cada step
        self._save_state()
        return self.exp


    def _save_state(self):
        """Guarda los archivos de estadísticas y resumen temporalmente."""
        self.stats.finish()
        
        stats_path = os.path.join(self.exp.stats_dir, "run_stats.json")
        self.stats.save(stats_path)
        self.exp.save_summary(
            stats_dict  = self.stats.to_dict(),
            seed_dict   = self.seed_mgr.to_dict(),
            config_dict = self.cfg,
        )


    # ── Retrocompatibilidad (One-Shot Run) ────────────────────────────────────
    
    def run(self, num_images: int | None = None) -> Experiment:
        """Modo clásico de ejecución One-Shot."""
        eng_cfg  = self.cfg.get("engine", {})
        n_images = int(num_images or eng_cfg.get("num_images", 50))
        
        self.setup()
        self.step(n_images)
        
        self._print_summary(self.exp, self.stats)
        return self.exp

    # ── Resumen consola ───────────────────────────────────────────────────────

    def _print_summary(self, exp: Experiment, stats: RunStatistics) -> None:
        d         = stats.to_dict()
        ok_color  = "green" if d["success_rate_pct"] >= 95 else "yellow"

        table = Table(
            title     = f"ASDGE RUN COMPLETED — {exp.experiment_id}",
            box       = box.DOUBLE_EDGE,
            border_style = "magenta",
            show_header  = False,
            padding      = (0, 2),
        )
        table.add_column("Metrica", style="bold white", width=32)
        table.add_column("Valor",   style="bold",       width=22)

        table.add_row("Experiment ID",   exp.experiment_id)
        table.add_row("Requested",       str(d["requested"]))
        table.add_row("Generated",       str(d["generated"]))
        table.add_row("Discarded",       str(d["discarded"]))
        table.add_row("Errors",          str(d["errors"]))
        table.add_row("Retries",         str(d["retries"]))
        table.add_row("Success Rate",    f"[{ok_color}]{d['success_rate_pct']} %[/{ok_color}]")
        table.add_row("Elapsed",         f"{d['elapsed_seconds']} s")

        tr_str = ", ".join(f"{k}:{v}" for k, v in d["transforms_applied"].items()) or "none"
        table.add_row("Transforms Used", tr_str)
        table.add_row("Output Dir",      exp.experiment_dir)

        console.print()
        console.print(table)
        console.print(Panel(
            f"[bold green]experiment_summary.json guardado en:[/bold green]\n{exp.summary_path}",
            border_style="green",
        ))
