"""
src/synthetic_generation/verifier/fingerprint.py
─────────────────────────────────────────────────
Calcula y genera el pipeline_fingerprint.json de un experimento.
Combina el fingerprint del dataset original (si existe), la configuración
y las transformaciones aplicadas.
"""

from __future__ import annotations
import os
import json
import hashlib
from typing import Dict, Any

class PipelineFingerprint:
    """Generador de hash único para certificar la reproducibilidad del pipeline."""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.summary_path   = os.path.join(experiment_dir, "experiment_summary.json")
        self.config_path    = os.path.join(experiment_dir, "config.yaml")

    def _hash_file(self, path: str) -> str:
        """Calcula SHA-256 de un archivo."""
        if not os.path.isfile(path):
            return ""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha.update(block)
        return sha.hexdigest()

    def generate(self) -> Dict[str, Any]:
        """Construye el fingerprint dict y lo guarda."""
        # Leer summary
        if not os.path.isfile(self.summary_path):
            raise FileNotFoundError(f"Falta {self.summary_path}")
        with open(self.summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        config_hash = self._hash_file(self.config_path)

        # Buscar el dataset fingerprint base si existe (Etapa 2.5)
        # Asumiendo que está en reports/dataset_fingerprint.json
        base_dir = os.path.abspath(os.path.join(self.experiment_dir, "..", ".."))
        base_fingerprint_path = os.path.join(base_dir, "reports", "dataset_fingerprint.json")
        base_fingerprint = ""
        if os.path.isfile(base_fingerprint_path):
            with open(base_fingerprint_path, "r", encoding="utf-8") as f:
                base_data = json.load(f)
                base_fingerprint = base_data.get("fingerprint", "")

        transforms = summary.get("run_statistics", {}).get("transforms_applied", {})
        seed = summary.get("reproducibility", {}).get("global_seed", 0)
        exp_id = summary.get("experiment_id", "")

        # Payload a hashear
        payload = {
            "experiment_id": exp_id,
            "pipeline_version": "1.0.0",
            "base_dataset_fingerprint": base_fingerprint,
            "config_hash": config_hash,
            "global_seed": seed,
            "transformations_used": list(transforms.keys())
        }

        # Calcular fingerprint final
        payload_str = json.dumps(payload, sort_keys=True)
        final_hash = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

        payload["pipeline_fingerprint"] = final_hash

        # Guardar en validation/
        out_dir = os.path.join(self.experiment_dir, "validation")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "pipeline_fingerprint.json")
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

        return payload
