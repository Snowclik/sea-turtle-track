import argparse
import sys
import os
from rich.console import Console

console = Console(force_terminal=True, highlight=False)

def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Synthetic Few-Shot Pipeline (ASFP)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    subparsers.add_parser("import",   help="Importar dataset oficial al proyecto.")
    subparsers.add_parser("analyze",  help="Analizar el dataset y generar reportes estadísticos.")
    subparsers.add_parser("validate", help="Validar integridad del dataset y generar DQI.")

    aug_parser = subparsers.add_parser("augment", help="Generar dataset sintético (Etapa 3).")
    aug_parser.add_argument("--config", default="configs/synthetic_generation.yaml",
                            help="Ruta al archivo de configuración YAML.")
    aug_parser.add_argument("--n", type=int, default=None,
                            help="Número de imágenes a generar (sobreescribe el config).")

    vrf_parser = subparsers.add_parser("verify", help="Validar experimento sintético (Etapa 3.5).")
    vrf_parser.add_argument("--exp", default="latest",
                            help="ID del experimento a verificar, o 'latest' para el más reciente.")

    exp_parser = subparsers.add_parser("experiment", help="Adaptive Synthetic Generation Framework (Etapa 4).")
    exp_subparsers = exp_parser.add_subparsers(dest="exp_command", help="Comandos de experimentación")
    exp_subparsers.add_parser("run", help="Ejecutar cola de experimentos (YAMLs en configs/experiments/)")
    exp_subparsers.add_parser("compare", help="Generar reporte comparativo de experimentos")
    
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark Framework (Etapa 5).")
    bench_subparsers = bench_parser.add_subparsers(dest="bench_command", help="Comandos de benchmark")
    bench_subparsers.add_parser("sync", help="Sincronizar experimentos al Leaderboard")
    bench_subparsers.add_parser("report", help="Generar reportes HTML/PDF/Excel del Leaderboard")

    parser_compare = subparsers.add_parser("compare", help="Ejecuta la herramienta de laboratorio para comparar modelos ONNX.")
    parser_compare.add_argument("--models", nargs="+", required=True, help="Rutas a los modelos ONNX a comparar")
    parser_compare.add_argument("--image", required=True, help="Ruta de la imagen a probar")
    parser_compare.add_argument("--out-dir", required=True, help="Directorio de salida para los reportes")

    train_parser = subparsers.add_parser("train", help="Ejecutar entrenamiento local (Etapa 6).")
    train_parser.add_argument("exp_id", help="Identificador del experimento (ej: baseline, geometric_only)")
    train_parser.add_argument("--resume", action="store_true", help="Reanudar el entrenamiento desde la última época truncada")

    tune_parser = subparsers.add_parser("tune", help="Hyperparameter Tuning (Etapa 10).")
    tune_parser.add_argument("exp_id", help="Identificador del experimento")
    tune_parser.add_argument("--iterations", type=int, default=30, help="Iteraciones genéticas")

    board_parser = subparsers.add_parser("board", help="Levantar TensorBoard local para monitoreo.")
    board_parser.add_argument("exp_id", help="Identificador del experimento")

    val_parser = subparsers.add_parser("validate-training", help="Validación Científica del Entrenamiento (Etapa 7.5).")
    val_parser.add_argument("exp_id", help="Identificador del experimento a validar (ej: baseline)")

    camp_parser = subparsers.add_parser("run-campaign", help="Ejecutar campaña masiva experimental (Etapa 8).")
    camp_parser.add_argument("campaign_id", help="ID de la campaña (ej: 001)")
    
    adapt_parser = subparsers.add_parser("adaptive", help="Ejecutar ASGF Adaptive Growth (Generación adaptativa + Fine-Tuning).")
    adapt_parser.add_argument("--config", default="configs/experiments/exp_d_synergy.yaml",
                              help="Ruta al archivo de configuración YAML.")

    inf_parser = subparsers.add_parser("infer", help="Realizar inferencia con el modelo (Etapa 9).")
    inf_parser.add_argument("target", help="Objetivo de inferencia (ej: external)")

    ds_parser = subparsers.add_parser("dataset", help="Dataset Expansion Framework (Etapa 11).")
    ds_parser.add_argument("action", choices=["diff", "rebuild", "import"], help="Acción a realizar en el dataset")
    ds_parser.add_argument("--input", help="Directorio de imágenes crudas para importar (solo para import)")

    subparsers.add_parser("evaluate", help="Evaluar modelo (Etapa 7).")
    subparsers.add_parser("report",   help="Generar reporte comparativo final (Etapa 8).")

    args = parser.parse_args()

    if args.command == "import":
        console.print("[bold yellow]Iniciando Dataset Importer...[/bold yellow]")
        sys.path.insert(0, ".")
        from utils.dataset_importer import run_import
        run_import()

    elif args.command == "analyze":
        console.print("[bold yellow]Iniciando Dataset Analyzer...[/bold yellow]")
        from analysis.dataset_analyzer import DatasetAnalyzer
        analyzer = DatasetAnalyzer()
        analyzer.analyze()
        analyzer.generate_figures()
        analyzer.generate_reports()

    elif args.command == "validate":
        console.print("[bold yellow]Iniciando Dataset Validator...[/bold yellow]")
        from analysis.dataset_validator import run_validation
        run_validation()

    elif args.command == "augment":
        engine_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_generation")
        sys.path.insert(0, engine_dir)
        from synthetic_generation.engine import SyntheticEngine
        engine = SyntheticEngine(config_path=args.config)
        engine.run(num_images=args.n)

    elif args.command == "verify":
        engine_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_generation")
        sys.path.insert(0, engine_dir)
        from synthetic_generation.verifier import ExperimentVerifier
        import glob
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        experiments_dir = os.path.join(base_dir, "experiments")
        
        if args.exp == "latest":
            exps = sorted(glob.glob(os.path.join(experiments_dir, "exp_*")), key=os.path.getmtime, reverse=True)
            if not exps:
                console.print("[bold red]No se encontraron experimentos en experiments/[/bold red]")
                sys.exit(1)
            exp_dir = exps[0]
        else:
            exp_dir = os.path.join(experiments_dir, args.exp)
            if not os.path.isdir(exp_dir):
                console.print(f"[bold red]Experimento no encontrado: {exp_dir}[/bold red]")
                sys.exit(1)
                
        verifier = ExperimentVerifier(exp_dir)
        verifier.run()
        
    elif args.command == "experiment":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from experiments import ASGFManager, ExperimentComparator
        
        configs_dir = os.path.join(base_dir, "configs", "experiments")
        exp_out_dir = os.path.join(base_dir, "experiments")
        
        if args.exp_command == "run":
            manager = ASGFManager(configs_dir, exp_out_dir)
            manager.run_batch()
            
        elif args.exp_command == "compare":
            comparator = ExperimentComparator(exp_out_dir)
            summaries = comparator.load_all_summaries()
            comparator.display_comparison(summaries)
            
        else:
            exp_parser.print_help()
            
    if args.command == "train":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        
        from training.trainer_manager import LocalTrainingManager
        manager = LocalTrainingManager(base_dir)
        try:
            manager.run_training(args.exp_id, resume=getattr(args, 'resume', False))
            console.print(f"[bold green]Entrenamiento completado para {args.exp_id}[/bold green]")
        except Exception as e:
            console.print(f"[bold red]Error durante el entrenamiento: {e}[/bold red]")
            import traceback
            traceback.print_exc()

    elif args.command == "tune":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        
        from training.tuner import YoloTuner
        tuner = YoloTuner(args.exp_id, base_dir)
        tuner.tune(args.iterations)

    elif args.command == "board":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from training.dataset_loader import DatasetLoader
        
        ds = DatasetLoader(args.exp_id, base_dir)
        tb_dir = os.path.join(ds.experiment_path, "training_results", "yolo_run")
        console.print(f"[bold cyan]Levantando TensorBoard en {tb_dir}...[/bold cyan]")
        console.print("[dim]Presiona Ctrl+C para salir.[/dim]")
        import subprocess
        subprocess.run(["tensorboard", f"--logdir={tb_dir}"])
            
    elif args.command == "benchmark":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from benchmark import BenchmarkManager
        
        manager = BenchmarkManager(base_dir)
        
        if args.bench_command == "sync":
            manager.sync()
        elif args.bench_command == "report":
            manager.report()
        else:
            bench_parser.print_help()
            
    elif args.command == "compare":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from tools.model_comparator.core.orchestrator import Orchestrator
        
        console.print("[bold yellow]Iniciando Universal Model Comparator...[/bold yellow]")
        Orchestrator.run_comparison(args.models, args.image, args.out_dir)
        
    elif args.command == "audit":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from verification import EcosystemAuditor
        
        auditor = EcosystemAuditor(base_dir)
        auditor.audit()

    elif args.command == "validate-training":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        
        from validation import TrainingValidator, TrainingReportGenerator
        val = TrainingValidator(args.exp_id, base_dir)
        if val.validate():
            rep = TrainingReportGenerator(args.exp_id, base_dir)
            rep.generate()
            console.print(f"[bold green]Validación científica exitosa para {args.exp_id}[/bold green]")
        else:
            console.print(f"[bold red]La validación falló para {args.exp_id}[/bold red]")

    elif args.command == "run-campaign":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        
        from benchmark.campaign_runner import CampaignRunner
        runner = CampaignRunner(args.campaign_id, base_dir)
        runner.run_all()
        
    elif args.command == "adaptive":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from experiments.growth_controller import GrowthController
        
        controller = GrowthController(args.config)
        controller.run()
        
    elif args.command == "infer":
        if args.target == "external":
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            sys.path.insert(0, os.path.join(base_dir, "src"))
            
            from inference import ExternalValidator
            val = ExternalValidator(base_dir)
            val.validate()
        else:
            console.print(f"[bold yellow]Objetivo de inferencia '{args.target}' no soportado.[/bold yellow]")

    elif args.command == "dataset":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from dataset_manager import DatasetManager
        
        manager = DatasetManager(base_dir)
        if args.action == "diff":
            manager.diff()
        elif args.action == "rebuild":
            manager.rebuild()
        elif args.action == "import":
            if not args.input:
                console.print("[bold red]⚠ Debes proveer un --input <directorio> para importar imágenes.[/bold red]")
            else:
                manager.import_data(args.input)

    elif args.command in ("evaluate", "report"):
        console.print(f"[bold yellow]Comando '{args.command}' aún no implementado (etapas futuras).[/bold yellow]")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
