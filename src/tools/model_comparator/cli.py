import argparse
from tools.model_comparator.core.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Universal ONNX Model Comparator")
    parser.add_argument("--models", nargs="+", required=True, help="List of model paths to compare")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--out-dir", required=True, help="Directory to save the reports")
    args = parser.parse_args()
    
    Orchestrator.run_comparison(args.models, args.image, args.out_dir)

if __name__ == "__main__":
    main()
