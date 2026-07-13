"""
src/synthetic_generation/transforms/__init__.py
Punto de entrada del paquete transforms.
Re-exporta todo lo necesario para que el resto del motor
solo necesite: from transforms import HorizontalFlip, ...
"""
from .base import Transform, Polygon
from .geometric   import HorizontalFlip, RandomRotate
from .photometric import BrightnessContrast

__all__ = [
    "Transform",
    "Polygon",
    "HorizontalFlip",
    "RandomRotate",
    "BrightnessContrast",
]
