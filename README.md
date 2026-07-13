# Adaptive Synthetic Few-Shot Pipeline (ASFP)

## Objetivo
Este proyecto busca diseñar una metodología reproducible para entrenar detectores de objetos cuando únicamente existe un conjunto reducido de imágenes reales. Demostraremos que un conjunto pequeño (como nuestras 31 imágenes originales de Playa Mata Oscura) combinado con generación sintética avanzada y transferencia de aprendizaje puede lograr un rendimiento competitivo.

## Filosofía
- **Modularidad:** Cada módulo resuelve un solo problema.
- **Reproducibilidad:** Los experimentos dependen de configuraciones de archivos YAML, sin hardcodear parámetros.
- **Escalabilidad:** Se pueden añadir nuevos módulos sin alterar los existentes.
- **Investigación:** Preparado para generar material publicable (gráficos, tablas, métricas).

## Arquitectura

El proyecto cuenta con un sistema de carpetas estructurado para experimentación, dividido en datos, configuración, código fuente (`src/`), modelos, y reportes.

- `dataset/`: Imágenes originales y sintéticas.
- `experiments/`: Seguimiento estructurado por cada iteración del experimento.
- `src/`: Código fuente modular.
- `configs/`: Centraliza todas las configuraciones.

## Flujo de Trabajo

La experimentación se maneja en múltiples iteraciones. La base (`exp01_baseline`) no utiliza aumentos drásticos. Las siguientes iteraciones incrementan gradualmente la generación sintética para comparar resultados de forma objetiva y aislada.

## Tecnologías
- Python, Albumentations, OpenCV, NumPy, Matplotlib, PyYAML.
