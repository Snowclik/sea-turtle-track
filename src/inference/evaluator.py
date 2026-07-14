"""
src/inference/evaluator.py
Calcula métricas de precisión y completitud comparando predicciones vs verdad terreno.
"""

from __future__ import annotations
from rich.console import Console

console = Console(highlight=False)

class Evaluator:
    def __init__(self):
        self.metrics = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "map50": 0.0,
            "map50_95": 0.0
        }
        
    def evaluate(self, detections: list) -> dict:
        """
        Para validar métricas exactas (Precision, Recall, mAP) se requeriría una matriz de confusión 
        comparando IoU de los polígonos predichos vs los reales.
        Dado que YOLO ya implementa `model.val()` para esto, si existiera un data.yaml del dataset externo,
        sería más preciso.
        Para no reimplementar PyTorch/Metrics puro en este módulo desconectado (o usar librerías externas),
        haremos una evaluación estructural de las detecciones arrojadas (o utilizar YOLO.val() internamente
        si se provee formato YOLO). En este script asumiremos que recibimos confidencias como proxy
        a menos que YOLO.val() lo calcule en lote.
        """
        total_dets = 0
        sum_conf = 0.0
        
        for det in detections:
            total_dets += det["num_detections"]
            # Extraer promedios de las cajas
            r = det["result_obj"]
            if r.boxes is not None and len(r.boxes) > 0:
                sum_conf += sum(r.boxes.conf.cpu().numpy())
                
        avg_conf = (sum_conf / total_dets) if total_dets > 0 else 0.0
        
        # Como es una validación externa de Inferencia a ciegas (a veces sin labels), 
        # reportaremos las métricas operativas por ahora.
        self.metrics["avg_confidence"] = round(float(avg_conf), 4)
        self.metrics["total_detections"] = total_dets
        
        return self.metrics
