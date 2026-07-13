"""
src/verification/ecosystem_checker.py
Audita los componentes lógicos del ecosistema.
"""

from __future__ import annotations
import os
import glob

class EcosystemChecker:
    def __init__(self, project_root: str):
        self.root = project_root
        self.results = {}

    def check_all(self) -> dict:
        self.results["Dataset"] = self._check_dataset()
        self.results["Configuration Files"] = self._check_configs()
        self.results["Benchmark Framework"] = self._check_benchmark()
        self.results["Experiments"] = self._check_experiments()
        self.results["Training Framework"] = self._check_training_framework()
        self.results["Fingerprints"] = self._check_fingerprints()
        return self.results

    def _check_dataset(self) -> bool:
        ds_orig = os.path.join(self.root, "dataset", "original")
        ds_lbls = os.path.join(self.root, "dataset", "labels")
        if not os.path.isdir(ds_orig) or not os.path.isdir(ds_lbls): return False
        
        # Original images might be jpg, png, jpeg
        import glob
        imgs = []
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG", "*.JPEG"):
            imgs.extend(glob.glob(os.path.join(ds_orig, ext)))
            
        lbls = glob.glob(os.path.join(ds_lbls, "*.txt"))
        if len(imgs) != 30 or len(lbls) != 30: return False
        return True

    def _check_configs(self) -> bool:
        req = ["augmentation.yaml", "synthetic_generation.yaml", "training_protocol.yaml"]
        cfg_dir = os.path.join(self.root, "configs")
        for r in req:
            if not os.path.isfile(os.path.join(cfg_dir, r)): return False
            if r == "training_protocol.yaml":
                try:
                    import yaml
                    with open(os.path.join(cfg_dir, r), "r", encoding="utf-8") as f:
                        yaml.safe_load(f)
                except Exception:
                    return False
        return True

    def _check_benchmark(self) -> bool:
        bench_dir = os.path.join(self.root, "benchmark_results")
        leaderboard = os.path.join(bench_dir, "benchmark_leaderboard.json")
        return os.path.isfile(leaderboard)

    def _check_experiments(self) -> bool:
        exp_dir = os.path.join(self.root, "experiments")
        exp_dirs = glob.glob(os.path.join(exp_dir, "exp_*"))
        if len(exp_dirs) < 1: return False
        
        for d in exp_dirs:
            if not os.path.isdir(os.path.join(d, "images")): return False
            if not os.path.isdir(os.path.join(d, "labels")): return False
            if not os.path.isfile(os.path.join(d, "experiment_summary.json")): return False
        return True

    def _check_training_framework(self) -> bool:
        import sys
        src_path = os.path.join(self.root, "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
            
        try:
            from training.trainer import YoloTrainer
            from training.dataset_loader import DatasetLoader
            from training.protocol_manager import TrainingProtocolAdapter
            from training.metrics_collector import MetricsCollector
            from training.exporter import ModelExporter
            from training.experiment_runner import TrainingRunner
            return True
        except ImportError:
            return False

    def _check_fingerprints(self) -> bool:
        # Check validation pipeline fp
        exp_dir = os.path.join(self.root, "experiments")
        val_summaries = glob.glob(os.path.join(exp_dir, "exp_*", "validation", "validation_summary.json"))
        has_val = len(val_summaries) > 0
        
        # Check training fp
        train_fps = glob.glob(os.path.join(exp_dir, "exp_*", "training_results", "training_fingerprint.json"))
        has_train = len(train_fps) > 0
        
        return has_val and has_train
