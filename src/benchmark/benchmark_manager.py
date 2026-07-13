"""
src/benchmark/benchmark_manager.py
Orquestador principal del Benchmark Framework.
"""

from __future__ import annotations
import os
from rich.console import Console

from .leaderboard import BenchmarkLeaderboard
from .benchmark_runner import BenchmarkRunner
from .benchmark_report import BenchmarkReporter

console = Console(force_terminal=True, highlight=False)

class BenchmarkManager:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.experiments_dir = os.path.join(project_root, "experiments")
        self.benchmark_dir = os.path.join(project_root, "benchmark_results")
        self.reports_dir = os.path.join(self.benchmark_dir, "reports")
        
        self.leaderboard = BenchmarkLeaderboard(self.benchmark_dir)

    def sync(self):
        """Sincroniza los experimentos generados al leaderboard."""
        console.print("[bold magenta]Iniciando sincronización del Benchmark...[/bold magenta]")
        runner = BenchmarkRunner(self.experiments_dir, self.leaderboard)
        synced = runner.sync()
        console.print(f"[bold green]Se han sincronizado {synced} experimentos en el Leaderboard.[/bold green]")

    def report(self):
        """Genera reportes de todos los formatos a partir del leaderboard."""
        console.print("[bold magenta]Generando reportes del Benchmark...[/bold magenta]")
        reporter = BenchmarkReporter(self.leaderboard.get_all(), self.reports_dir)
        
        json_path = reporter.generate_json()
        console.print(f"[green]✔ JSON:[/green] {json_path}")
        
        html_path = reporter.generate_html()
        console.print(f"[green]✔ HTML:[/green] {html_path}")
        
        pdf_path = reporter.generate_pdf()
        console.print(f"[green]✔ PDF:[/green] {pdf_path}")
        
        excel_path = reporter.generate_excel()
        if excel_path:
            console.print(f"[green]✔ EXCEL:[/green] {excel_path}")
        else:
            console.print("[yellow]⚠ EXCEL: omitido (openpyxl no instalado)[/yellow]")
