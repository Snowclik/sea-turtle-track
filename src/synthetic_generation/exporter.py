"""
src/synthetic_generation/exporter.py
─────────────────────────────────────
Única responsabilidad: persistir todos los artefactos de una
imagen sintética generada:
  - Imagen (JPG/PNG)
  - Etiqueta YOLO Segmentation (.txt)
  - JSON de metadatos (delegado a metadata.ImageMetadata.save)
"""

from __future__ import annotations
import os
from typing import List, Tuple

import cv2
import numpy as np

from metadata import ImageMetadata

Polygon = List[Tuple[float, float]]


class Exporter:
    """
    Exporta una imagen sintética y su etiqueta YOLO Seg al sistema de archivos.
    No toma decisiones sobre qué exportar — solo persiste lo que recibe.
    """

    def __init__(
        self,
        output_images_dir:   str,
        output_labels_dir:   str,
        output_metadata_dir: str,
        output_format:       str = "jpg",
        output_quality:      int = 95,
    ):
        self.images_dir   = output_images_dir
        self.labels_dir   = output_labels_dir
        self.metadata_dir = output_metadata_dir
        self.format       = output_format.lower()
        self.quality      = output_quality

        os.makedirs(self.images_dir,   exist_ok=True)
        os.makedirs(self.labels_dir,   exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

    # ── Imagen ────────────────────────────────────────────────────────────────

    def save_image(self, image: np.ndarray, filename: str) -> str:
        """Guarda la imagen. Devuelve la ruta completa."""
        path = os.path.join(self.images_dir, filename)
        if self.format == "jpg":
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        else:
            cv2.imwrite(path, image)
        return path

    # ── Etiqueta YOLO Seg ─────────────────────────────────────────────────────

    def save_label(
        self,
        polygons:  List[Polygon],
        class_ids: List[int],
        filename:  str,
    ) -> str:
        """
        Guarda la etiqueta en formato YOLO Segmentation:
            <class_id> x1 y1 x2 y2 ... xN yN

        Devuelve la ruta completa.
        """
        stem  = os.path.splitext(filename)[0]
        path  = os.path.join(self.labels_dir, f"{stem}.txt")
        lines = []
        for cls_id, poly in zip(class_ids, polygons):
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            lines.append(f"{cls_id} {coords}")
        with open(path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        return path

    # ── Metadatos ─────────────────────────────────────────────────────────────

    def save_metadata(self, meta: ImageMetadata) -> str:
        """Delega a ImageMetadata.save y devuelve la ruta del JSON."""
        return meta.save(self.metadata_dir)

    # ── Exportación completa ──────────────────────────────────────────────────

    def export(
        self,
        image:     np.ndarray,
        polygons:  List[Polygon],
        class_ids: List[int],
        filename:  str,
        meta:      ImageMetadata,
    ) -> dict:
        """
        Exporta imagen, etiqueta y metadatos en una sola llamada.
        Devuelve las rutas generadas.
        """
        img_path  = self.save_image(image, filename)
        lbl_path  = self.save_label(polygons, class_ids, filename)
        meta_path = self.save_metadata(meta)
        return {
            "image":    img_path,
            "label":    lbl_path,
            "metadata": meta_path,
        }
