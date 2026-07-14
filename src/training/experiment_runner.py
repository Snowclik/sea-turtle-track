"""
src/training/experiment_runner.py
Orquesta el entrenamiento individual de un experimento.
"""

from __future__ import annotations
import os
import json
from rich.console import Console

from .dataset_loader import DatasetLoader
from .protocol_manager import TrainingProtocolAdapter
from .model_factory import ModelFactory
from .trainer import YoloTrainer

console = Console(force_terminal=True, highlight=False)

class TrainingRunner:
    def __init__(self, experiment_id: str, project_root: str):
        self.experiment_id = experiment_id
        self.project_root = project_root
        
    def run(self, resume: bool = False):
        console.print(f"\n[bold cyan]Iniciando LTF (Local Training Framework) para: {self.experiment_id}[/bold cyan]")
        if resume:
            console.print("  [bold yellow]⚠ MODO REANUDACIÓN ACTIVADO (Intentará recuperar el último last.pt)[/bold yellow]")
        
        # 1. Cargar Dataset
        console.print("  [dim]>> Resolviendo dataset...[/dim]")
        ds = DatasetLoader(self.experiment_id, self.project_root)
        ds_info = ds.prepare_for_training()
        console.print(f"  [green]✔ Dataset resuelto:[/green] {os.path.basename(ds_info['experiment_path'])}")
        
        # 2. Cargar Protocolo Experimental
        console.print("  [dim]>> Cargando Protocolo Experimental estricto...[/dim]")
        protocol = TrainingProtocolAdapter()
        cfg = protocol.get_config()
        console.print(f"  [green]✔ Protocolo cargado:[/green] versión {cfg.get('version')}")
        
        # 3. Preparar Modelo
        console.print("  [dim]>> Instanciando ModelFactory...[/dim]")
        arch = cfg.get("model", {}).get("architecture", "unknown")
        factory = ModelFactory(architecture=arch, device=cfg.get("hardware", {}).get("device", "auto"))
        model = factory.build()
        console.print(f"  [green]✔ Modelo preparado:[/green] {arch}")
        
        # 4. Crear carpeta de salida del entrenamiento
        train_out = os.path.join(ds_info["experiment_path"], "training_results")
        os.makedirs(train_out, exist_ok=True)
        
        # 5. Generar Fingerprint
        console.print("  [dim]>> Generando Training Fingerprint...[/dim]")
        # Extraer fingerprints previos del validation_summary.json si existe
        val_summary_path = os.path.join(ds_info["experiment_path"], "validation", "validation_summary.json")
        base_ds_fp = "unknown"
        pipe_fp = "unknown"
        if os.path.isfile(val_summary_path):
            with open(val_summary_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
                pipe_fp = val_data.get("pipeline_fingerprint", "unknown")
                # Intentar sacar el del dataset original del experiment_summary
                exp_summary_path = os.path.join(ds_info["experiment_path"], "experiment_summary.json")
                if os.path.isfile(exp_summary_path):
                    with open(exp_summary_path, "r", encoding="utf-8") as fs:
                        exp_data = json.load(fs)
                        base_ds_fp = exp_data.get("reproducibility", {}).get("global_seed", "unknown")

        protocol.generate_fingerprint(
            exp_id=os.path.basename(ds_info["experiment_path"]),
            output_dir=train_out,
            dataset_fp=base_ds_fp,
            pipeline_fp=pipe_fp
        )
        console.print(f"  [green]✔ Fingerprint sellado.[/green]")
        
        console.print("  [dim]>> Iniciando entrenamiento...[/dim]")
        trainer = YoloTrainer(model, cfg, ds_info, train_out)
        metrics = trainer.train(resume=resume)
        
        # 7. Guardar métricas
        summary_path = os.path.join(train_out, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        
        console.print(f"  [green]✔ Training Summary guardado:[/green] {os.path.basename(summary_path)}")
        console.print(f"\n[bold green]Entrenamiento completado para {self.experiment_id}[/bold green]\n")
