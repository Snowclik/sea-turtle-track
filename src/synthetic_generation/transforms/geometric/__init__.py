"""
src/synthetic_generation/transforms/geometric/__init__.py
Exporta todas las transformaciones geométricas disponibles.
Para añadir una nueva: importarla aquí y registrarla en registry.py.
"""
from .horizontal_flip import HorizontalFlip
from .random_rotate   import RandomRotate

__all__ = ["HorizontalFlip", "RandomRotate"]
