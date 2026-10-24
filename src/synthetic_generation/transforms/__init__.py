"""
src/synthetic_generation/transforms/__init__.py
Punto de entrada del paquete transforms.
Re-exporta todo lo necesario para que el resto del motor
solo necesite: from transforms import HorizontalFlip, ...
"""
from .base import Transform, Polygon
from .geometric   import HorizontalFlip, RandomRotate, RandomScale, RandomPerspective
from .photometric import BrightnessContrast, MotionBlur

__all__ = [
    "Transform",
    "Polygon",
    "HorizontalFlip",
    "RandomRotate",
    "RandomScale",
    "RandomPerspective",
    "BrightnessContrast",
    "MotionBlur",
]
