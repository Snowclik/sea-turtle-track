"""
src/protocol/fingerprint.py
Genera el training_fingerprint.json para un entrenamiento ejecutado.
"""

from __future__ import annotations
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, Any

from .protocol_manager import ProtocolManager

class TrainingFingerprint:
    """Generador del fingerprint científico de un entrenamiento."""

    def __init__(self, experiment_id: str, output_dir: str):
        self.experiment_id = experiment_id
        self.output_dir = output_dir
        self.protocol = ProtocolManager()

    def generate(self, dataset_fingerprint: str, pipeline_fingerprint: str) -> Dict[str, Any]:
        """Genera y guarda el fingerprint sellado."""
        cfg = self.protocol.get_config()
        
        payload = {
            "experiment_id": self.experiment_id,
            "dataset_fingerprint": dataset_fingerprint,
            "pipeline_fingerprint": pipeline_fingerprint,
            "protocol_version": self.protocol.get_version(),
            "protocol_hash": self.protocol.hash,
            "model_architecture": cfg.get("model", {}).get("architecture", "unknown"),
            "training_seed": cfg.get("training", {}).get("seed", 0),
            "execution_date": datetime.now().isoformat(timespec="seconds")
        }
        
        payload_str = json.dumps(payload, sort_keys=True)
        final_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        
        payload["training_fingerprint"] = final_hash
        
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, "training_fingerprint.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
            
        return payload
