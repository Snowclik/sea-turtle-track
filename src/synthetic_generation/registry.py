"""
src/synthetic_generation/registry.py
─────────────────────────────────────
Única responsabilidad: mantener el mapa nombre → clase de transformación.

Patrón Registry: las transformaciones se registran por su nombre
y pueden instanciarse dinámicamente desde la configuración YAML.

Uso:
    from registry import TransformRegistry
    transform = TransformRegistry.build("HorizontalFlip", probability=0.5)
"""

from __future__ import annotations
from typing import Dict, Type, Any

from transforms import Transform, HorizontalFlip, RandomRotate, BrightnessContrast


class TransformRegistry:
    """
    Registro central de transformaciones disponibles.
    Para registrar una nueva transformación, añadirla al dict _REGISTRY.
    """

    _REGISTRY: Dict[str, Type[Transform]] = {
        "HorizontalFlip":    HorizontalFlip,
        "RandomRotate":      RandomRotate,
        "BrightnessContrast": BrightnessContrast,
    }

    @classmethod
    def build(cls, name: str, **kwargs) -> Transform:
        """
        Instancia una transformación por nombre.
        Los kwargs se pasan al constructor de la clase.
        """
        if name not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise KeyError(f"Transformación '{name}' no registrada. Disponibles: {available}")
        return cls._REGISTRY[name](**kwargs)

    @classmethod
    def available(cls) -> list:
        return list(cls._REGISTRY.keys())

    @classmethod
    def register(cls, name: str, klass: Type[Transform]) -> None:
        """Permite registrar nuevas transformaciones en tiempo de ejecución."""
        if not issubclass(klass, Transform):
            raise TypeError(f"{klass} debe ser subclase de Transform")
        cls._REGISTRY[name] = klass
