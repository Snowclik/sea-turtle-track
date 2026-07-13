"""
src/synthetic_generation/registry.py
──────────────────────────────────────
Única responsabilidad: mantener el mapa nombre → clase de transformación.

Para registrar una nueva transformación:
  1. Crear su archivo en transforms/geometric/ o transforms/photometric/
  2. Importarla en el __init__.py correspondiente
  3. Añadirla al dict _REGISTRY aquí

El motor (engine.py) NO necesita modificarse.
"""

from __future__ import annotations
from typing import Dict, List, Type, Any

from transforms import Transform, HorizontalFlip, RandomRotate, BrightnessContrast


class TransformRegistry:
    """
    Registro central de transformaciones disponibles en el motor ASDGE.

    Diseñado para extensión sin modificación (Open/Closed Principle):
    nuevas transformaciones se registran aquí sin tocar el motor.
    """

    _REGISTRY: Dict[str, Type[Transform]] = {
        # ── Geométricas ───────────────────────────────────────────────────────
        "HorizontalFlip":     HorizontalFlip,
        "RandomRotate":       RandomRotate,

        # ── Fotométricas ──────────────────────────────────────────────────────
        "BrightnessContrast": BrightnessContrast,

        # ── Añadir nuevas transformaciones aquí ───────────────────────────────
        # "VerticalFlip":     VerticalFlip,
        # "GaussianNoise":    GaussianNoise,
        # "RandomCrop":       RandomCrop,
    }

    @classmethod
    def build(cls, name: str, **kwargs) -> Transform:
        """
        Instancia una transformación por nombre con los parámetros del config.
        Lanza KeyError si el nombre no está registrado.
        """
        if name not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise KeyError(
                f"Transformación '{name}' no registrada. "
                f"Disponibles: {available}"
            )
        return cls._REGISTRY[name](**kwargs)

    @classmethod
    def register(cls, name: str, klass: Type[Transform]) -> None:
        """
        Registra una nueva transformación en tiempo de ejecución.
        Útil para transformaciones externas o de terceros.
        """
        if not issubclass(klass, Transform):
            raise TypeError(f"{klass.__name__} debe ser subclase de Transform")
        cls._REGISTRY[name] = klass

    @classmethod
    def available(cls) -> List[str]:
        return sorted(cls._REGISTRY.keys())

    @classmethod
    def by_category(cls) -> Dict[str, List[str]]:
        """Agrupa los nombres de transformaciones por categoría."""
        result: Dict[str, List[str]] = {}
        for name, klass in cls._REGISTRY.items():
            cat = klass().category if not klass.__abstractmethods__ else "unknown"
            result.setdefault(cat, []).append(name)
        return result
