# Adaptive Synthetic Few-Shot Pipeline (ASFP)
## Plan Maestro de Desarrollo
### Etapa 1 — Arquitectura Base del Proyecto

---

# Visión del Proyecto

Este proyecto no consiste únicamente en entrenar un modelo de detección de objetos.

El propósito es diseñar una metodología reproducible para entrenar detectores de objetos cuando únicamente existe un conjunto reducido de imágenes reales.

La investigación busca demostrar que un conjunto pequeño de imágenes cuidadosamente seleccionadas, combinado con generación sintética avanzada y transferencia de aprendizaje, puede producir un rendimiento competitivo frente a conjuntos de entrenamiento mucho mayores.

Toda la arquitectura deberá diseñarse pensando en una futura publicación científica, por lo que la reproducibilidad, documentación y modularidad son requisitos obligatorios.

---

# Objetivo de la Etapa 1

Construir la infraestructura completa del proyecto.

**No se entrenará ningún modelo.**

**No se generarán imágenes todavía.**

**No se utilizará YOLO.**

**No se utilizará Roboflow.**

El único objetivo de esta etapa es dejar preparada una arquitectura profesional que permita desarrollar posteriormente todo el pipeline de investigación.

---

# Dataset Inicial

El proyecto utilizará inicialmente:

- 31 imágenes originales capturadas en Playa Mata Oscura.
- Sus respectivas anotaciones en formato YOLO.

Las imágenes originales son el activo más importante del proyecto.

Nunca deberán modificarse.

Nunca deberán sobrescribirse.

Todo procesamiento deberá generar nuevos archivos.

---

# Filosofía del Desarrollo

Todo el código deberá cumplir los siguientes principios.

## Modularidad

Cada módulo debe resolver únicamente un problema.

No crear archivos enormes.

No mezclar responsabilidades.

---

## Reproducibilidad

Todo experimento deberá poder repetirse utilizando únicamente un archivo de configuración.

No deben existir parámetros escritos directamente dentro del código.

---

## Escalabilidad

La arquitectura deberá permitir agregar nuevos módulos sin necesidad de modificar los existentes.

---

## Investigación

Cada etapa deberá producir resultados reutilizables posteriormente dentro del artículo científico.

---

# Tecnologías

Lenguaje principal

- Python

Editor

- Visual Studio Code

Control de versiones

- Git

Bibliotecas

- Albumentations
- OpenCV
- NumPy
- Matplotlib
- PyYAML
- tqdm

---

# Arquitectura del Proyecto

```

adaptive-synthetic-fewshot/

│

├── dataset/

│ ├── original/

│ ├── labels/

│ ├── synthetic/

│ ├── validation/

│ ├── test/

│ └── preview/

│

├── experiments/

│ ├── exp01_baseline/

│ ├── exp02_aug500/

│ ├── exp03_aug1000/

│ ├── exp04_aug3000/

│ └── exp05_aug5000/

│

├── paper/

│ ├── figures/

│ ├── tables/

│ ├── diagrams/

│ └── drafts/

│

├── reports/

├── logs/

├── notebooks/

├── models/

│

├── configs/

│ └── augmentation.yaml

│

├── src/

│ ├── synthetic_generation/

│ │ ├── augment.py

│ │ ├── pipeline.py

│ │ └── config.py

│ │

│ ├── analysis/

│ │ ├── dataset_stats.py

│ │ ├── compare_experiments.py

│ │ ├── plot_metrics.py

│ │ └── confusion_matrix.py

│ │

│ ├── visualization/

│ ├── utils/

│ └── main.py

│

├── requirements.txt

├── README.md

└── .gitignore

```

---

# Tareas de la Etapa 1

## 1. Crear la arquitectura del proyecto

Construir exactamente la estructura anterior.

No agregar carpetas innecesarias.

---

## 2. Configurar el entorno

Crear un entorno virtual.

Instalar únicamente:

- albumentations
- opencv-python
- numpy
- matplotlib
- pyyaml
- tqdm

Generar automáticamente `requirements.txt`.

---

## 3. Preparar el dataset

Copiar las 31 imágenes originales dentro de:

```

dataset/original/

```

Copiar todas las etiquetas YOLO dentro de:

```

dataset/labels/

```

No modificar absolutamente ningún archivo.

---

## 4. Construir el módulo de análisis del dataset

Crear `dataset_stats.py`.

Este módulo únicamente analizará el dataset.

No modificará imágenes.

Deberá calcular automáticamente:

- cantidad de imágenes
- resolución promedio
- resolución mínima y máxima
- ancho promedio
- alto promedio
- tamaño promedio
- número de objetos
- cantidad de bounding boxes
- porcentaje de ocupación
- distribución espacial del objeto
- orientación de imágenes
- estadísticas generales

Toda la información deberá exportarse posteriormente a un reporte.

---

## 5. Sistema de configuración

Toda la configuración deberá almacenarse en:

```

configs/augmentation.yaml

```

Nunca utilizar valores fijos dentro del código.

---

## 6. Preparar el motor de generación sintética

Crear la arquitectura del módulo `synthetic_generation`.

Todavía no implementar lógica.

Debe quedar preparado para soportar posteriormente:

- Albumentations
- Transformaciones geométricas
- Transformaciones fotométricas
- Escenarios sintéticos
- Fondos sintéticos
- Exportación automática

---

## 7. Sistema de visualización

Preparar la arquitectura del módulo `visualization`.

Posteriormente deberá permitir visualizar:

Imagen original

↓

Imagen sintética

↓

Bounding Boxes

↓

Transformaciones aplicadas

No implementar todavía.

---

## 8. Inicializar Git

Realizar el primer commit.

Mensaje sugerido:

```

Initial research architecture

```

---

## 9. README

Crear un README profesional.

Debe explicar:

- objetivo del proyecto
- filosofía
- estructura
- arquitectura
- flujo de trabajo
- tecnologías
- licencia

---

# Restricciones

Durante esta etapa está prohibido:

- entrenar modelos
- utilizar YOLO
- utilizar Roboflow
- utilizar ONNX
- utilizar React
- generar datasets sintéticos
- realizar inferencias

La única finalidad es construir una arquitectura sólida y preparada para las siguientes etapas.

---

# Resultado Esperado

Al finalizar esta etapa el proyecto deberá estar completamente organizado, documentado y preparado para comenzar el desarrollo del **Adaptive Synthetic Few-Shot Pipeline (ASFP)**.

No debe existir lógica experimental todavía.

El repositorio debe representar la base de una investigación científica reproducible y escalable.