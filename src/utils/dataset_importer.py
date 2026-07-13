"""
src/utils/dataset_importer.py
Importa el dataset desde la ruta de descarga hacia la estructura del proyecto.
No modifica los archivos originales — únicamente copia.
"""

import os
import shutil
import glob
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich import box

console = Console(force_terminal=True, highlight=False)

SOURCE_IMAGES_DIR = r"C:\Users\tavog\Downloads\turtle-trak.yolo26\train\images"
SOURCE_LABELS_DIR = r"C:\Users\tavog\Downloads\turtle-trak.yolo26\train\labels"

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEST_IMAGES = os.path.join(BASE_DIR, "dataset", "original")
DEST_LABELS = os.path.join(BASE_DIR, "dataset", "labels")

VALID_IMG_EXT = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')


def _clean_dest(dest_dir):
    """Elimina archivos existentes en el destino para garantizar limpieza."""
    removed = 0
    for f in glob.glob(os.path.join(dest_dir, "*")):
        if os.path.isfile(f):
            os.remove(f)
            removed += 1
    return removed


def run_import():
    console.print("\n[bold yellow]Iniciando Dataset Importer...[/bold yellow]")

    # Verificar origen
    if not os.path.isdir(SOURCE_IMAGES_DIR):
        console.print(f"[bold red]ERROR: No se encontró el directorio de imágenes origen:[/bold red]\n  {SOURCE_IMAGES_DIR}")
        return False
    if not os.path.isdir(SOURCE_LABELS_DIR):
        console.print(f"[bold red]ERROR: No se encontró el directorio de etiquetas origen:[/bold red]\n  {SOURCE_LABELS_DIR}")
        return False

    # Crear destinos
    os.makedirs(DEST_IMAGES, exist_ok=True)
    os.makedirs(DEST_LABELS, exist_ok=True)

    # Limpiar destinos previos
    console.print("[bold cyan]>> Limpiando dataset previo en el proyecto...[/bold cyan]")
    removed_imgs   = _clean_dest(DEST_IMAGES)
    removed_labels = _clean_dest(DEST_LABELS)
    console.print(f"   Eliminados: {removed_imgs} imágenes previas, {removed_labels} etiquetas previas.")

    # Obtener listas origen
    src_images = sorted([
        p for p in glob.glob(os.path.join(SOURCE_IMAGES_DIR, "*"))
        if p.endswith(VALID_IMG_EXT)
    ])
    src_labels = sorted(glob.glob(os.path.join(SOURCE_LABELS_DIR, "*.txt")))

    # ── Copiar imágenes ────────────────────────────────────────────────────────
    console.print(f"[bold cyan]>> Copiando {len(src_images)} imágenes...[/bold cyan]")
    copied_imgs = 0
    for src_path in track(src_images, description="  Imágenes"):
        fname = os.path.basename(src_path)
        dest  = os.path.join(DEST_IMAGES, fname)
        shutil.copy2(src_path, dest)
        copied_imgs += 1

    # ── Copiar etiquetas ───────────────────────────────────────────────────────
    console.print(f"[bold cyan]>> Copiando {len(src_labels)} etiquetas...[/bold cyan]")
    copied_lbls = 0
    for src_path in track(src_labels, description="  Etiquetas"):
        fname = os.path.basename(src_path)
        dest  = os.path.join(DEST_LABELS, fname)
        shutil.copy2(src_path, dest)
        copied_lbls += 1

    # ── Resumen ────────────────────────────────────────────────────────────────
    table = Table(title="DATASET IMPORT COMPLETED", box=box.DOUBLE_EDGE,
                  border_style="green", show_header=False, padding=(0, 2))
    table.add_column("Item",  style="bold white", width=30)
    table.add_column("Valor", style="bold green", width=20)
    table.add_row("Source", SOURCE_IMAGES_DIR)
    table.add_row("Images copied",  str(copied_imgs))
    table.add_row("Labels copied",  str(copied_lbls))
    table.add_row("Destination img", DEST_IMAGES)
    table.add_row("Destination lbl", DEST_LABELS)

    console.print()
    console.print(table)
    return True
