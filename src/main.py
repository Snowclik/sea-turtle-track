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

    train_parser = subparsers.add_parser("train", help="Entrenar modelo (Etapa 6).")
    train_parser.add_argument("exp_id", help="Identificador del experimento (ej: baseline, exp_2026...)")
    
    subparsers.add_parser("audit", help="Auditoría completa del ecosistema (Etapa 6.5).")

    val_parser = subparsers.add_parser("validate-training", help="Validación Científica del Entrenamiento (Etapa 7.5).")
    val_parser.add_argument("exp_id", help="Identificador del experimento a validar (ej: baseline)")

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

    elif args.command == "train":
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, os.path.join(base_dir, "src"))
        from training import LocalTrainingManager
        
        manager = LocalTrainingManager(base_dir)
        manager.train_experiment(args.exp_id)
        
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

    elif args.command in ("evaluate", "report"):
        console.print(f"[bold yellow]Comando '{args.command}' aún no implementado (etapas futuras).[/bold yellow]")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
