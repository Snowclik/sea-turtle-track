"""
src/synthetic_generation/transforms/photometric/__init__.py
Exporta todas las transformaciones fotométricas disponibles.
Para añadir una nueva: importarla aquí y registrarla en registry.py.
"""
from .brightness_contrast import BrightnessContrast
from .motion_blur import MotionBlur

__all__ = ["BrightnessContrast", "MotionBlur"]
