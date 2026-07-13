"""
src/training/dataset_loader.py
Identifica y prepara las rutas del dataset de un experimento generado.
"""

from __future__ import annotations
import os
import glob

class DatasetLoader:
    def __init__(self, experiment_identifier: str, project_root: str):
        """
        experiment_identifier puede ser un ID exacto (exp_2026...) 
        o el prefijo del config (ej: baseline, geometric_only).
        """
        self.project_root = project_root
        self.identifier = experiment_identifier
        self.experiments_dir = os.path.join(project_root, "experiments")
        self.experiment_path = self._resolve_path()

    def _resolve_path(self) -> str:
        """Resuelve el path real del experimento en base al identificador."""
        # Intento 1: match exacto por nombre de carpeta
        exact_path = os.path.join(self.experiments_dir, self.identifier)
        if os.path.isdir(exact_path):
            return exact_path
            
        # Intento 2: buscar el exp más reciente generado por ese config name (ej: 'baseline')
        pattern = os.path.join(self.experiments_dir, "exp_*")
        dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        
        for d in dirs:
            summary_path = os.path.join(d, "experiment_summary.json")
            if os.path.isfile(summary_path):
                import json
                with open(summary_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    cfg_name = data.get("config_file", "").replace(".yaml", "")
                    if cfg_name == self.identifier:
                        return d
                        
        raise FileNotFoundError(f"No se encontró dataset asociado al experimento '{self.identifier}'")

    def prepare_for_training(self) -> dict:
        """Devuelve las rutas necesarias para YOLO (YAML simulado o real)."""
        images_dir = os.path.join(self.experiment_path, "images")
        labels_dir = os.path.join(self.experiment_path, "labels")
        
        if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"El dataset {self.experiment_path} está incompleto o corrupto.")
            
        return {
            "experiment_path": self.experiment_path,
            "images": images_dir,
            "labels": labels_dir
        }
