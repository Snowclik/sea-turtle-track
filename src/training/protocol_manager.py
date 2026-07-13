"""
src/training/protocol_manager.py
Adaptador para consumir el Protocolo Experimental único.
"""

from __future__ import annotations
import os
import sys

# Inyectamos el root para acceder a src.protocol si es necesario
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(base_dir, "src") not in sys.path:
    sys.path.insert(0, os.path.join(base_dir, "src"))

from protocol import ProtocolManager as CoreProtocolManager
from protocol import TrainingFingerprint

class TrainingProtocolAdapter:
    def __init__(self):
        self._manager = CoreProtocolManager()
        
    def get_config(self) -> dict:
        return self._manager.get_config()
        
    def get_hash(self) -> str:
        return self._manager.hash
        
    def generate_fingerprint(self, exp_id: str, output_dir: str, dataset_fp: str, pipeline_fp: str) -> dict:
        fp = TrainingFingerprint(exp_id, output_dir)
        return fp.generate(dataset_fp, pipeline_fp)
