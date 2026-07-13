"""
src/synthetic_generation/experiment.py
────────────────────────────────────────
Única responsabilidad: gestionar la identidad y estructura de un
experimento de generación sintética.

Cada ejecución del motor genera:
  - Un identificador único: exp_YYYYMMDD_HHMMSS_XXXX
  - Una carpeta de experimento dentro de experiments/
  - Una copia del archivo de configuración utilizado
  - Un experiment_summary.json con el resumen completo
"""

from __future__ import annotations
import os
import json
import shutil
import string
import uuid
from datetime import datetime
from typing import Any, Dict


class Experiment:
    """
    Identidad, rutas y artefactos de un experimento de generación.

    Estructura de carpetas generada:
        experiments/
            exp_YYYYMMDD_HHMMSS_XXXX/
                images/
                labels/
                previews/
                metadata/
                stats/
                config.yaml          ← copia del config utilizado
                experiment_summary.json
    """

    SUBDIRS = ["images", "labels", "previews", "metadata", "stats"]

    def __init__(self, base_experiments_dir: str, config_path: str):
        self.config_path        = config_path
        self.base_dir           = base_experiments_dir
        self.experiment_id      = self._generate_id()
        self.experiment_dir     = os.path.join(base_experiments_dir, self.experiment_id)
        self.created_at         = datetime.now().isoformat(timespec="seconds")
        self._summary: Dict[str, Any] = {}

        # Crear estructura de carpetas
        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(self.experiment_dir, sub), exist_ok=True)

        # Copiar config al experimento
        self._copy_config()

    # ── Rutas públicas ────────────────────────────────────────────────────────

    @property
    def images_dir(self) -> str:
        return os.path.join(self.experiment_dir, "images")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.experiment_dir, "labels")

    @property
    def previews_dir(self) -> str:
        return os.path.join(self.experiment_dir, "previews")

    @property
    def metadata_dir(self) -> str:
        return os.path.join(self.experiment_dir, "metadata")

    @property
    def stats_dir(self) -> str:
        return os.path.join(self.experiment_dir, "stats")

    @property
    def config_copy_path(self) -> str:
        return os.path.join(self.experiment_dir, "config.yaml")

    @property
    def summary_path(self) -> str:
        return os.path.join(self.experiment_dir, "experiment_summary.json")

    # ── Generación de ID ──────────────────────────────────────────────────────

    @staticmethod
    def _generate_id() -> str:
        """
        Genera un ID único y legible:
            exp_YYYYMMDD_HHMMSS_XXXX
        donde XXXX son 4 caracteres alfanuméricos aleatorios.
        """
        now    = datetime.now()
        suffix = uuid.uuid4().hex[:4].upper()
        return f"exp_{now.strftime('%Y%m%d_%H%M%S')}_{suffix}"

    # ── Config ───────────────────────────────────────────────────────────────

    def _copy_config(self) -> None:
        if os.path.isfile(self.config_path):
            shutil.copy2(self.config_path, self.config_copy_path)

    # ── Resumen ───────────────────────────────────────────────────────────────

    def save_summary(
        self,
        stats_dict:  Dict[str, Any],
        seed_dict:   Dict[str, Any],
        config_dict: Dict[str, Any],
    ) -> str:
        """
        Genera y guarda el experiment_summary.json.
        Devuelve la ruta del archivo guardado.
        """
        self._summary = {
            "experiment_id":  self.experiment_id,
            "created_at":     self.created_at,
            "finished_at":    datetime.now().isoformat(timespec="seconds"),
            "config_file":    os.path.basename(self.config_path),
            "config":         config_dict,
            "reproducibility": seed_dict,
            "run_statistics": stats_dict,
            "output_dir":     self.experiment_dir,
        }
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(self._summary, f, indent=4, ensure_ascii=False)
        return self.summary_path

    def to_dict(self) -> Dict[str, Any]:
        return self._summary
