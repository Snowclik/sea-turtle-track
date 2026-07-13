"""
src/verification/system_checker.py
Audita el entorno físico del sistema.
"""

from __future__ import annotations
import os
import sys
import platform
import psutil
import shutil

class SystemChecker:
    def __init__(self, project_root: str):
        self.root = project_root
        self.results = {}
        
    def check_all(self) -> dict:
        sys_status = True
        
        # Python >= 3.8
        py_ver = sys.version_info
        if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 8):
            sys_status = False
            
        # x64
        if "64" not in platform.architecture()[0]:
            sys_status = False
            
        # RAM (> 4GB total)
        mem = psutil.virtual_memory()
        if mem.total < 4 * 1024**3:
            sys_status = False
            
        # Disco (> 5GB free)
        disk = shutil.disk_usage(self.root)
        if disk.free < 5 * 1024**3:
            sys_status = False
            
        # Permisos
        try:
            test_file = os.path.join(self.root, "reports", "verification", ".test")
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("test")
            os.remove(test_file)
        except Exception:
            sys_status = False
            
        # UTF-8
        if sys.getdefaultencoding().lower() != "utf-8":
            # Just a warning or we can be strict
            pass

        self.results["System Requirements"] = sys_status
        return self.results
