"""
src/protocol/protocol_manager.py
Lee y valida el protocolo de entrenamiento. Calcula el hash.
"""

from __future__ import annotations
import os
import yaml
import json
import hashlib
from typing import Dict, Any

class ProtocolManager:
    """Clase encargada de cargar y asegurar el protocolo de entrenamiento único."""
    
    def __init__(self, protocol_path: str = "configs/training_protocol.yaml"):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.protocol_path = os.path.join(base_dir, protocol_path)
        self.config = self._load()
        self.hash = self._compute_hash()

    def _load(self) -> Dict[str, Any]:
        if not os.path.isfile(self.protocol_path):
            raise FileNotFoundError(f"No se encontró el protocolo de entrenamiento en {self.protocol_path}")
        with open(self.protocol_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _compute_hash(self) -> str:
        """Calcula el SHA-256 del contenido del archivo."""
        sha = hashlib.sha256()
        with open(self.protocol_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha.update(block)
        return sha.hexdigest()

    def get_version(self) -> str:
        return str(self.config.get("version", "unknown"))

    def get_config(self) -> Dict[str, Any]:
        """Devuelve un diccionario con toda la configuración."""
        return self.config
