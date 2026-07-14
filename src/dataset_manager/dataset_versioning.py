"""
src/dataset_manager/dataset_versioning.py
Mantiene un registro maestro inmutable de las versiones del dataset a medida que crece.
"""

from __future__ import annotations
import os
import json
import time
from datetime import datetime
from rich.console import Console

console = Console(highlight=False)

class DatasetVersioner:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.versions_dir = os.path.join(project_root, "dataset", ".versions")
        self.registry_file = os.path.join(self.versions_dir, "versions_registry.json")
        
        os.makedirs(self.versions_dir, exist_ok=True)
        self._init_registry()
        
    def _init_registry(self):
        if not os.path.isfile(self.registry_file):
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump({"versions": []}, f, indent=4)
                
    def get_current_version(self) -> int:
        with open(self.registry_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            vlist = data.get("versions", [])
            return len(vlist)
            
    def get_history(self) -> list:
        with open(self.registry_file, "r", encoding="utf-8") as f:
            return json.load(f).get("versions", [])
            
    def get_latest_stats(self) -> dict:
        v = self.get_history()
        if not v:
            return {"images": 0, "objects": 0, "fingerprint": None}
        return v[-1]

    def create_version(self, num_images: int, num_objects: int, fingerprint: str) -> dict:
        """Sella una nueva versión en el registry."""
        history = self.get_history()
        next_v = len(history) + 1
        
        v_data = {
            "version": f"v{next_v}",
            "date": datetime.now().isoformat(),
            "images": num_images,
            "objects": num_objects,
            "fingerprint": fingerprint
        }
        
        history.append(v_data)
        
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump({"versions": history}, f, indent=4)
            
        console.print(f"[green]✔ Dataset versión {v_data['version']} sellada ({num_images} imgs, {num_objects} objs).[/green]")
        return v_data
