"""
src/benchmark/campaign_runner.py
Orquesta la ejecución masiva de la campaña experimental ASGF.
"""

from __future__ import annotations
import os
import sys
import subprocess
from rich.console import Console

console = Console(highlight=False)

class CampaignRunner:
    def __init__(self, campaign_id: str, project_root: str):
        self.campaign_id = campaign_id
        self.project_root = project_root
        self.campaign_dir = os.path.join(project_root, "campaigns", f"Campaign_{campaign_id}")
        os.makedirs(self.campaign_dir, exist_ok=True)
        
        # El orden requerido científicamente
        self.experiments = ["baseline", "geometric_only", "photometric_only", "mixed"]

    def run_all(self):
        console.print(f"\n[bold magenta]=====================================================[/bold magenta]")
        console.print(f"[bold magenta] INICIANDO CAMPAÑA EXPERIMENTAL ASGF: Campaign {self.campaign_id}[/bold magenta]")
        console.print(f"[bold magenta]=====================================================[/bold magenta]\n")
        
        main_py = os.path.join(self.project_root, "src", "main.py")
        
        for exp in self.experiments:
            console.print(f"\n[bold cyan]▶ Iniciando evaluación científica para: {exp}[/bold cyan]")
            
            # 1. Training
            console.print(f"  [dim]>> Paso 1/3: Entrenando modelo (esto puede tomar tiempo)...[/dim]")
            subprocess.run([sys.executable, main_py, "train", exp], cwd=self.project_root, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            
            # 2. Validation
            console.print(f"  [dim]>> Paso 2/3: Validando inferencia y consolidando métricas...[/dim]")
            subprocess.run([sys.executable, main_py, "validate-training", exp], cwd=self.project_root, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            
            # 3. Sincronizar Benchmark
            console.print(f"  [dim]>> Paso 3/3: Sincronizando Leaderboard...[/dim]")
            subprocess.run([sys.executable, main_py, "benchmark", "sync"], cwd=self.project_root, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            
        console.print(f"\n[bold green]✔ Todos los experimentos de la campaña han finalizado la ejecución física.[/bold green]")
        
        # Generar Reporte de la Campaña y Seleccionar OFFICIAL MODEL
        from .campaign_report import CampaignReportGenerator
        reporter = CampaignReportGenerator(self.campaign_id, self.project_root)
        reporter.generate_and_crown()
