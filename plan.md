# Etapa 2 - Dataset Analyzer

## Objetivo

Desarrollar un módulo denominado **Dataset Analyzer** que analice automáticamente el conjunto de imágenes originales y genere un reporte completo del dataset sin modificar ningún archivo.

El análisis debe ser completamente automático.

No debe requerir intervención manual.

No debe modificar las imágenes.

No debe modificar las etiquetas.

Únicamente debe analizar y generar reportes.

---

# Entrada

dataset/

    original/

    labels/

---

# Salidas

reports/

    dataset_report.html

    dataset_report.pdf

    dataset_statistics.json

    dataset_statistics.csv

paper/

    figures/

    paper/tables/

---

# Información a extraer

## Información general

- Número total de imágenes
- Número total de etiquetas
- Número total de objetos
- Clases existentes
- Imágenes sin etiquetas
- Etiquetas sin imagen

---

## Resoluciones

- Resolución mínima
- Resolución máxima
- Resolución promedio

Crear un histograma.

---

## Tamaño de archivos

- mínimo
- máximo
- promedio

---

## Bounding Boxes

Calcular automáticamente

- cantidad total
- ancho promedio
- alto promedio
- área promedio
- porcentaje ocupado
- posición promedio

Generar histogramas.

---

## Distribución espacial

Crear un heatmap mostrando dónde aparecen normalmente los rastros.

---

## Estadísticas por imagen

Para cada imagen generar

- nombre
- resolución
- tamaño
- cantidad de objetos
- área ocupada

Guardar en CSV.

---

## Análisis de color

Calcular automáticamente

- brillo promedio
- contraste
- histograma RGB
- histograma HSV

---

## EXIF

Si existen metadatos, extraer automáticamente

- fecha
- hora
- GPS
- altitud
- cámara
- fabricante
- modelo
- orientación

Si un campo no existe colocar

N/A

---

## Visualizaciones

Generar automáticamente

- histogramas
- gráficos de barras
- gráficos circulares
- heatmap de bounding boxes

Guardar todas las figuras en

paper/figures/

---

## Reporte HTML

Crear un reporte moderno con

- resumen
- tablas
- gráficos
- estadísticas

---

## Reporte PDF

Generar automáticamente un PDF profesional utilizando la información del reporte HTML.

---

## Exportación

Exportar automáticamente

dataset_statistics.json

dataset_statistics.csv

---

# Restricciones

No utilizar IA.

No modificar imágenes.

No modificar etiquetas.

No generar imágenes sintéticas.

No entrenar modelos.

Este módulo es únicamente un analizador del dataset.

---

# Objetivo científico

Este reporte será la línea base del proyecto.

Posteriormente, el mismo módulo deberá poder analizar el dataset sintético para comparar ambos conjuntos y verificar si la generación sintética conserva una distribución visual coherente.


Una mejora que haría desde ahora

En lugar de llamar al archivo dataset_stats.py, lo renombraría a:

dataset_analyzer.py