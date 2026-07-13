"""
src/benchmark/comparator.py
Lógica de comparación entre experimentos basada en el Leaderboard.
"""

from __future__ import annotations
from typing import Dict, Any, List

class BenchmarkComparator:
    def __init__(self, leaderboard_data: Dict[str, Any]):
        self.data = leaderboard_data

    def get_ranked_experiments(self, metric: str = "date", reverse: bool = True) -> List[tuple[str, dict]]:
        """Devuelve experimentos ordenados por la métrica indicada. 
        Por defecto ordena por fecha más reciente."""
        items = list(self.data.items())
        
        if metric == "date":
            items.sort(key=lambda x: x[1]["metadata"]["date"], reverse=reverse)
        elif metric == "mAP_50":
            # Ordena los que tengan mAP, los demás al fondo
            items.sort(key=lambda x: (x[1]["training_metrics"].get("mAP_50") is not None, 
                                      x[1]["training_metrics"].get("mAP_50", 0)), 
                       reverse=reverse)
        return items
