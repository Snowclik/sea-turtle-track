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

    subparsers.add_parser("train",    help="Entrenar modelo (Etapa 4).")
    subparsers.add_parser("evaluate", help="Evaluar modelo (Etapa 5).")
    subparsers.add_parser("report",   help="Generar reporte comparativo (Etapa 6).")

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

    elif args.command in ("train", "evaluate", "report"):
        console.print(f"[bold yellow]Comando '{args.command}' aún no implementado (etapas futuras).[/bold yellow]")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
