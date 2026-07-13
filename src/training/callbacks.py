"""
src/training/callbacks.py
Hooks de entrenamiento para YOLO26.
(Esqueleto).
"""

from __future__ import annotations

class TrainingCallbacks:
    """Administra los eventos durante el entrenamiento."""
    
    def on_train_start(self):
        pass
        
    def on_epoch_end(self, epoch: int, metrics: dict):
        pass
        
    def on_train_end(self):
        pass
