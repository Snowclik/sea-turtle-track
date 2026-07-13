"""
src/training/trainer_manager.py
Punto de entrada global para el framework de entrenamiento.
"""

from __future__ import annotations
import os
from .experiment_runner import TrainingRunner

class LocalTrainingManager:
    def __init__(self, project_root: str):
        self.project_root = project_root
        
    def train_experiment(self, experiment_id: str):
        """Prepara e inicia el runner individual."""
        runner = TrainingRunner(experiment_id, self.project_root)
        runner.run()
