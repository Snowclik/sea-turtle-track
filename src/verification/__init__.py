"""
src/verification/__init__.py
"""
from .ecosystem_checker import EcosystemChecker
from .system_checker import SystemChecker
from .report_generator import ReportGenerator

class EcosystemAuditor:
    def __init__(self, project_root: str):
        self.root = project_root
        
    def audit(self):
        eco = EcosystemChecker(self.root)
        eco_results = eco.check_all()
        
        sys = SystemChecker(self.root)
        sys_results = sys.check_all()
        
        report = ReportGenerator(self.root, eco_results, sys_results)
        report.display_console()
        report.export_all()
