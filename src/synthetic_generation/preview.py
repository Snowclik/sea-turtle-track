"""
src/synthetic_generation/preview.py
─────────────────────────────────────
Única responsabilidad: generar imágenes comparativas (side-by-side)
que muestran la imagen original y la sintética con sus polígonos dibujados.

Formato de salida:
  [ Imagen Original + Polígono ] | [ Imagen Sintética + Polígono Transformado ]
  Barra inferior: nombre del archivo + transformaciones aplicadas
"""

from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np

Polygon = List[Tuple[float, float]]

# Colores BGR
COLOR_ORIGINAL  = (0, 230, 64)    # verde neón
COLOR_SYNTHETIC = (80, 160, 255)  # azul claro
COLOR_TEXT      = (255, 255, 255)
COLOR_BAR       = (20, 20, 20)


class PreviewGenerator:
    """Genera imágenes comparativas para inspección visual del pipeline."""

    def __init__(self, output_dir: str, max_width: int = 1280):
        self.output_dir = output_dir
        self.max_width  = max_width
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self,
        original_image:   np.ndarray,
        synthetic_image:  np.ndarray,
        original_polygons:  List[Polygon],
        synthetic_polygons: List[Polygon],
        transforms_applied: List[Dict[str, Any]],
        output_filename:  str,
    ) -> str:
        """
        Genera y guarda la imagen comparativa.
        Devuelve la ruta del archivo guardado.
        """
        orig   = self._draw_polygons(original_image.copy(),  original_polygons,  COLOR_ORIGINAL,  "ORIGINAL")
        synth  = self._draw_polygons(synthetic_image.copy(), synthetic_polygons, COLOR_SYNTHETIC, "SYNTHETIC")

        # Redimensionar ambas al mismo alto
        orig,  synth = self._equalize_heights(orig, synth)

        # Concatenar horizontalmente
        combined = np.hstack([orig, synth])

        # Añadir barra inferior con transformaciones
        combined = self._add_info_bar(combined, transforms_applied)

        # Escalar si supera el ancho máximo
        h, w = combined.shape[:2]
        if w > self.max_width * 2:
            scale    = (self.max_width * 2) / w
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)))

        path = os.path.join(self.output_dir, output_filename)
        cv2.imwrite(path, combined)
        return path

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _draw_polygons(
        self,
        image:    np.ndarray,
        polygons: List[Polygon],
        color:    Tuple[int, int, int],
        label:    str,
    ) -> np.ndarray:
        h, w = image.shape[:2]

        for poly in polygons:
            if len(poly) < 3:
                continue
            pts = np.array(
                [[int(x * w), int(y * h)] for x, y in poly],
                dtype=np.int32,
            )
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=3)
            overlay = image.copy()
            cv2.fillPoly(overlay, [pts], color)
            image = cv2.addWeighted(overlay, 0.18, image, 0.82, 0)

        # Etiqueta de encabezado
        cv2.rectangle(image, (0, 0), (w, 44), COLOR_BAR, -1)
        cv2.putText(image, label, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        return image

    def _equalize_heights(self, img_a: np.ndarray, img_b: np.ndarray):
        ha, hb = img_a.shape[0], img_b.shape[0]
        if ha == hb:
            return img_a, img_b
        target = min(ha, hb)
        def _resize_h(img, target_h):
            h, w = img.shape[:2]
            scale = target_h / h
            return cv2.resize(img, (int(w * scale), target_h))
        return _resize_h(img_a, target), _resize_h(img_b, target)

    def _add_info_bar(
        self,
        image: np.ndarray,
        transforms_applied: List[Dict[str, Any]],
    ) -> np.ndarray:
        bar_h = 36
        bar   = np.full((bar_h, image.shape[1], 3), COLOR_BAR, dtype=np.uint8)
        names = [t["name"] for t in transforms_applied if t.get("applied", False)]
        text  = "Transforms: " + (", ".join(names) if names else "none")
        cv2.putText(bar, text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 1, cv2.LINE_AA)
        return np.vstack([image, bar])
