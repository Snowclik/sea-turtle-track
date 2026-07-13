"""
src/synthetic_generation/metadata.py
─────────────────────────────────────
Única responsabilidad: construir y serializar el JSON de trazabilidad
para cada imagen sintética generada.

Formato de salida:
{
  "source_image": "...",
  "generated_image": "...",
  "seed": 123456,
  "width": 5472,
  "height": 3648,
  "n_polygons": 2,
  "transformations": [...],
  "validation": {"status": "valid", "issues": []},
  "timestamp": "2026-07-13T21:00:00"
}
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any
import json
import os
from datetime import datetime


@dataclass
class TransformRecord:
    """Registro de una única transformación aplicada."""
    name: str
    parameters: Dict[str, Any]
    applied: bool   # False si la prob no se cumplió

    def to_dict(self) -> Dict:
        return {
            "name":       self.name,
            "parameters": self.parameters,
            "applied":    self.applied,
        }


@dataclass
class ImageMetadata:
    """Trazabilidad completa de una imagen sintética."""
    source_image:    str
    generated_image: str
    seed:            int
    width:           int
    height:          int
    n_polygons:      int
    transformations: List[TransformRecord] = field(default_factory=list)
    validation:      Dict[str, Any]        = field(default_factory=lambda: {"status": "valid", "issues": []})
    timestamp:       str                   = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))

    def add_transform(self, record: TransformRecord) -> None:
        self.transformations.append(record)

    def set_validation(self, status: str, issues: List[str]) -> None:
        self.validation = {"status": status, "issues": issues}

    def to_dict(self) -> Dict:
        return {
            "source_image":    self.source_image,
            "generated_image": self.generated_image,
            "seed":            self.seed,
            "width":           self.width,
            "height":          self.height,
            "n_polygons":      self.n_polygons,
            "transformations": [t.to_dict() for t in self.transformations],
            "validation":      self.validation,
            "timestamp":       self.timestamp,
        }

    def save(self, output_dir: str) -> str:
        """Serializa y guarda el JSON. Devuelve la ruta del archivo."""
        stem   = os.path.splitext(self.generated_image)[0]
        fname  = f"{stem}.json"
        fpath  = os.path.join(output_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4, ensure_ascii=False)
        return fpath
