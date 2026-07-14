"""
src/dataset_manager/__init__.py
Punto de entrada global para el Dataset Expansion Framework (DEF).
"""

from __future__ import annotations
import os
import sys
from rich.console import Console

from .dataset_versioning import DatasetVersioner
from .dataset_merger import DatasetMerger
from .dataset_diff import DatasetDiff

console = Console(highlight=False)

class DatasetManager:
    def __init__(self, project_root: str):
        self.project_root = project_root
        
    def import_data(self, input_dir: str):
        merger = DatasetMerger(self.project_root)
        merger.merge(input_dir)
        
    def diff(self):
        differ = DatasetDiff(self.project_root)
        differ.diff()
        
    def rebuild(self):
        console.print(f"\n[bold magenta]=====================================================[/bold magenta]")
        console.print(f"[bold magenta] DATASET EXPANSION FRAMEWORK: REBUILDING ASGF [/bold magenta]")
        console.print(f"[bold magenta]=====================================================[/bold magenta]\n")
        
        # 1. Chequear Diff
        differ = DatasetDiff(self.project_root)
        changes = differ.diff()
        
        if not changes:
            console.print("[yellow]⚠ Reconstrucción abortada. No hay cambios.[/yellow]")
            return
            
        console.print("\n[bold cyan]1/5 Ejecutando Validación DQI...[/bold cyan]")
        import subprocess
        main_py = os.path.join(self.project_root, "src", "main.py")
        subprocess.run([sys.executable, main_py, "validate"], cwd=self.project_root)
        
        console.print("\n[bold cyan]2/5 Re-generando Análisis Estadístico...[/bold cyan]")
        subprocess.run([sys.executable, main_py, "analyze"], cwd=self.project_root)
        
        console.print("\n[bold cyan]3/5 Re-generando Datasets Sintéticos (ASGF)...[/bold cyan]")
        # Aquí reconstruiremos el baseline, geometric, photometric y mixed.
        # Para hacer esto, ASGF toma el config de experiment. Le delegamos a los scripts.
        # El comando 'augment' usa el generation_protocol.yaml, o podemos usar `experiment` loop.
        # Para simplificar y respetar el ASGF, llamaremos 'experiment' con un script maestro si existe, 
        # pero es mejor llamar "augment" o ejecutar el framework de Benchmark ASGF sync.
        # Ejecutamos las etapas de la campaña pero a nivel de regeneración sintética:
        for ex in ["baseline", "geometric_only", "photometric_only", "mixed"]:
            subprocess.run([sys.executable, main_py, "experiment", ex], cwd=self.project_root)
            
        console.print("\n[bold cyan]4/5 Sincronizando Benchmark...[/bold cyan]")
        subprocess.run([sys.executable, main_py, "benchmark", "sync"], cwd=self.project_root)
        
        console.print("\n[bold cyan]5/5 Sellando nueva versión de Dataset...[/bold cyan]")
        versioner = DatasetVersioner(self.project_root)
        # Fingerprint simulado o extraído del DQI
        import hashlib
        fp = hashlib.md5(f"v_{changes['current_img']}_{changes['current_obj']}".encode()).hexdigest()
        
        v_data = versioner.create_version(
            num_images=changes["current_img"],
            num_objects=changes["current_obj"],
            fingerprint=fp
        )
        
        console.print(f"\n[bold green]✔ El dataset ha sido reconstruido exitosamente en la {v_data['version']}[/bold green]")
