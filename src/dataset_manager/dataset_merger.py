"""
src/dataset_manager/dataset_merger.py
Importador y fusionador de datos. Toma una carpeta cruda, la filtra y la inyecta al dataset original.
"""

from __future__ import annotations
import os
import glob
import shutil
from rich.console import Console

from .quality_filter import QualityFilter
from .duplicate_detector import DuplicateDetector
from .dataset_versioning import DatasetVersioner

console = Console(highlight=False)

class DatasetMerger:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.dataset_dir = os.path.join(project_root, "dataset", "original")
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.filter = QualityFilter()
        self.detector = DuplicateDetector(self.dataset_dir, threshold=5)
        self.versioner = DatasetVersioner(project_root)
        
    def merge(self, input_dir: str):
        console.print(f"\n[bold magenta]=====================================================[/bold magenta]")
        console.print(f"[bold magenta] DATASET EXPANSION FRAMEWORK: IMPORTANDO [/bold magenta]")
        console.print(f"[bold magenta]=====================================================[/bold magenta]\n")
        
        if not os.path.isdir(input_dir):
            console.print(f"[red]✖ El directorio crudo {input_dir} no existe.[/red]")
            return
            
        # 1. Leer candidatos
        candidatos = glob.glob(os.path.join(input_dir, "*.*"))
        exts = [".jpg", ".jpeg", ".png"]
        candidatos = [c for c in candidatos if os.path.splitext(c)[1].lower() in exts]
        
        if not candidatos:
            console.print(f"[yellow]⚠ No se encontraron imágenes en {input_dir}.[/yellow]")
            return
            
        console.print(f"  [dim]>> Analizando {len(candidatos)} candidatos...[/dim]")
        
        aprobados = 0
        rechazados_calidad = 0
        rechazados_duplicados = 0
        
        # 2. Filtrar e inyectar
        for c in candidatos:
            if not self.filter.is_valid(c):
                rechazados_calidad += 1
                continue
                
            if self.detector.is_duplicate(c):
                rechazados_duplicados += 1
                continue
                
            # Aprobar
            basename = os.path.basename(c)
            dst_img = os.path.join(self.dataset_dir, basename)
            
            # Mover/Copiar imagen
            shutil.copy(c, dst_img)
            
            # Buscar label asociado
            name_no_ext = os.path.splitext(basename)[0]
            label_src = os.path.join(input_dir, f"{name_no_ext}.txt")
            if os.path.isfile(label_src):
                shutil.copy(label_src, os.path.join(self.dataset_dir, f"{name_no_ext}.txt"))
                
            # Añadir al detector online
            self.detector.existing_hashes[dst_img] = self.detector.existing_hashes.get(dst_img)
            aprobados += 1
            
        console.print(f"\n[bold green]✔ Ingestión completada.[/bold green]")
        console.print(f"  + [green]{aprobados} imágenes nuevas aceptadas.[/green]")
        if rechazados_duplicados > 0:
            console.print(f"  - [yellow]{rechazados_duplicados} rechazadas por duplicación (pHash).[/yellow]")
        if rechazados_calidad > 0:
            console.print(f"  - [red]{rechazados_calidad} rechazadas por calidad o corrupción.[/red]")
            
        # 3. Forzar reconstrucción si hubo cambios
        if aprobados > 0:
            console.print("\n[bold cyan]Cambios detectados. Usa `python src/main.py dataset rebuild` para sellar la nueva versión y sincronizar todo el ASGF.[/bold cyan]")
