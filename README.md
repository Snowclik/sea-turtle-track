# 🐢 Sea Turtle Track — Sistema Inteligente Basado en UAV y YOLO26 para la Detección de Rastros de Anidación de Tortugas Marinas

<div align="center">

![Kempis Turtle Tracker Banner](https://img.shields.io/badge/Sea_Turtle_Track-Investigaci%C3%B3n_Cient%C3%ADfica_%26_Conservaci%C3%B3n-1A8A4A?style=for-the-badge&logo=target)

**Monitoreo aéreo costero autónomo mediante Vehículos Aéreos No Tripulados (UAV) y Visión por Computador de Ultra Baja Latencia en CPU**

[![JIC UTP 2026](https://img.shields.io/badge/🥉_3.er_Lugar-JIC_UTP_2026_(Ingenier%C3%ADa)-FFD700?style=for-the-badge)](https://utp.ac.pa)
[![JIC Nacional IESTEC 2026](https://img.shields.io/badge/🏆_Finalista_Nacional-JIC_Nacional_2026-005CED?style=for-the-badge)](https://utp.ac.pa)

[![Tauri v2](https://img.shields.io/badge/Tauri-v2.0-24C8DB?style=flat-square&logo=tauri&logoColor=white)](https://tauri.app)
[![Rust](https://img.shields.io/badge/Backend-Rust_1.79+-DEA584?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![React](https://img.shields.io/badge/Frontend-React_18_%2B_TS-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![pnpm](https://img.shields.io/badge/Package_Manager-pnpm-F69220?style=flat-square&logo=pnpm&logoColor=white)](https://pnpm.io)
[![ONNX Runtime](https://img.shields.io/badge/Inferencia-ONNX_Runtime_CPU-005CED?style=flat-square&logo=onnx&logoColor=white)](https://onnxruntime.ai)
[![YOLO26-n](https://img.shields.io/badge/Arquitectura-YOLO26--n_(Ultra_Light)-FF4B4B?style=flat-square&logo=yolo&logoColor=white)](https://github.com/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
[![Dataset Ganador](https://img.shields.io/badge/Roboflow_Universe-Dataset_Ganador_Oficial-9B51E0?style=flat-square&logo=roboflow&logoColor=white)](https://universe.roboflow.com/snowcliks-workspace/turtle-track-detector-2-205gp)
[![Licencia](https://img.shields.io/badge/Licencia-Acad%C3%A9mica_%2B_Atribuci%C3%B3n-green?style=flat-square)](LICENSE)

</div>

---

## 📌 Tabla de Contenidos

1. [Reconocimientos Científicos y Contexto Institucional](#1-reconocimientos-científicos-y-contexto-institucional)
2. [Descripción General, Misión Biológica y Contexto de Campo](#2-descripción-general-misión-biológica-y-contexto-de-campo)
3. [Especies de Tortugas Marinas y Características de sus Rastros](#3-especies-de-tortugas-marinas-y-características-de-sus-rastros)
4. [¿Por qué UAVs (Drones)? Justificación de la Perspectiva Aérea](#4-por-qué-uavs-drones-justificación-de-la-perspectiva-aérea)
5. [Física del Sustrato: Arena Oscura (Mata Oscura) vs Arenas Claras (Europa/Mediterráneo)](#5-física-del-sustrato-arena-oscura-mata-oscura-vs-arenas-claras-europamediterráneo)
6. [Bases de Datos y Evolución Histórica de los Datasets](#6-bases-de-datos-y-evolución-histórica-de-los-datasets)
7. [¿Por qué NO Machine Learning Tradicional vs Deep Learning?](#7-por-qué-no-machine-learning-tradicional-vs-deep-learning)
8. [¿Qué es YOLO26 y por qué la Variante YOLO26-n? (ProgLoss, STAL y NMS-Free)](#8-qué-es-yolo26-y-por-qué-la-variante-yolo26-n-progloss-stal-y-nms-free)
9. [¿Por qué ONNX Runtime y NO Inferencia Directa con Python?](#9-por-qué-onnx-runtime-y-no-inferencia-directa-con-python)
10. [Arquitectura de Software: ¿Por qué Tauri (Rust + React + Vite + Tailwind)?](#10-arquitectura-de-software-por-qué-tauri-rust--react--vite--tailwind)
11. [Entrenamiento del Modelo: Local vs Google Colab](#11-entrenamiento-del-modelo-local-vs-google-colab)
12. [Validación Científica de Hipótesis y Resultados Estadísticos](#12-validación-científica-de-hipótesis-y-resultados-estadísticos)
13. [Desempeño según Iluminación y Matriz Exhaustiva de Limitaciones](#13-desempeño-según-iluminación-y-matriz-exhaustiva-de-limitaciones)
14. [Estudio Social y Percepción Comunitaria (Alineación ODS)](#14-estudio-social-y-percepción-comunitaria-alineación-ods)
15. [Requisitos de Hardware y Software](#15-requisitos-de-hardware-y-software)
16. [Guía Paso a Paso para Reproducir el Proyecto](#16-guía-paso-a-paso-para-reproducir-el-proyecto)
17. [Estructura del Repositorio](#17-estructura-del-repositorio)
18. [Propiedad Intelectual, Delimitación de Autoría y Citación Obligatoria](#18-propiedad-intelectual-delimitación-de-autoría-y-citación-obligatoria)
19. [Agradecimientos y Licencia](#19-agradecimientos-y-licencia)

---

## 1. Reconocimientos Científicos y Contexto Institucional

El presente proyecto de investigación aplicada fue desarrollado bajo el marco de la **Universidad Tecnológica de Panamá (UTP)** en colaboración con organizaciones de conservación comunitaria. Los resultados fueron presentados y evaluados en las competencias de investigación formativa de pregrado más exigentes de la República de Panamá:

```
                            ┌───────────────────────────────────────────────┐
                            │    JORNADA DE INICIACIÓN CIENTÍFICA (JIC)     │
                            └──────────────────────┬────────────────────────┘
                                                   │
                       ┌───────────────────────────┴───────────────────────────┐
                       ▼                                                       ▼
        ┌─────────────────────────────┐                         ┌─────────────────────────────┐
        │        JIC UTP 2026         │                         │      JIC NACIONAL 2026      │
        │   Fase Institucional UTP    │                         │    Fase Nacional SENACYT    │
        ├─────────────────────────────┤                         ├─────────────────────────────┤
        │ • 🥉 3.er Lugar             │                         │ • 🏆 FINALISTA NACIONAL     │
        │ • Categoría: Ingeniería     │                         │ • Avalado por SENACYT / UTP │
        │ • Evaluación entre todas    │                         │ • Certamen científico       │
        │   las facultades de la UTP  │                         │   nacional de Panamá        │
        └─────────────────────────────┘                         └─────────────────────────────┘
```

### Distinción Oficial: JIC UTP vs JIC Nacional (SENACYT / UTP)
- **JIC UTP 2026 (Fase Institucional):** Certamen de investigación científica de pregrado que congrega a todas las facultades de la **Universidad Tecnológica de Panamá**. El proyecto obtuvo el **3.er Lugar Institucional en la Categoría de Ingeniería**, destacándose por su rigor experimental, solidez estadística y viabilidad tecnológica en campo.
- **JIC Nacional 2026 (Fase Nacional):** Máxima instancia de investigación formativa universitaria en Panamá, coorganizada y financiada por la **Secretaría Nacional de Ciencia, Tecnología e Innovación (SENACYT)** y el Consejo de Rectores. En este certamen compiten exclusivamente los mejores proyectos clasificados de todas las universidades públicas y particulares del país. La fase final se celebra en el marco del **IESTEC** (*Congreso Internacional de Ingeniería, Ciencias y Tecnología*). El proyecto fue galardonado con la distinción oficial de **Finalista Nacional en la Categoría de Ingeniería y Tecnología**.

---

## 2. Descripción General, Misión Biológica y Contexto de Campo

Las tortugas marinas son especies ecológicamente fundamentales para el equilibrio de las cadenas tróficas y el mantenimiento de los ecosistemas marino-costeros. Sin embargo, sus poblaciones sufren un declive alarmante por pérdida de hábitat, saqueo furtivo de nidos, depredadores invasores y captura incidental.

### Área de Estudio y Operación: Playa Mata Oscura
- **Ubicación Geográfica:** Distrito de Mariato, Provincia de Veraguas, República de Panamá (costa del Pacífico).
- **Extensión Costera:** Aproximadamente **5.0 km** de playa ininterrumpida.
- **Esfuerzo de Patrullaje Comunitario:** El monitoreo lo lidera la **Fundación Agua y Tierra**. El equipo cuenta con únicamente **cuatro personas** que rotan la vigilancia nocturna, saliendo habitualmente **dos voluntarios por noche** (uno a cada extremo de la playa).
- **Cobertura Humana Limitada:** En una jornada nocturna de dos horas a pie, dos voluntarios logran cubrir únicamente **3.5 km de los 5.0 km totales**, dejando 1.5 km desprotegidos.
- **Pico Reproductivo:** Entre **septiembre y noviembre**, cuando se llegan a registrar hasta **8 tortugas anidando en una misma noche** (frente al promedio habitual de 1 a 2), la capacidad humana queda completamente desbordada.
- **Temporada Completa de Anidación:** Se extiende de **junio a marzo**, registrándose un promedio global de **150 nidos por temporada**.

```
  Playa Mata Oscura (5.0 km de frente costero)
 ├─────────────────────────────────────── 3.5 km ───────────────────────────────────────┤───── 1.5 km ─────┤
 │ Patrullaje a pie (2 voluntarios / noche) — Cobertura lenta y fatiga física           │ ZONA NO CUBIERTA │
 ├──────────────────────────────────────────────────────────────────────────────────────┴──────────────────┤
 │ Monitoreo con Dron UAV + YOLO26 (25.4 FPS) — COBERTURA TOTAL DE LOS 5.0 KM EN 15 MINUTOS                │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Alcance Biológico Exclusivo:** El sistema está enfocado y calibrado **únicamente para tortugas marinas** y sus rastros de anidación en la arena. No está diseñado para fauna terrestre general ni mamíferos en aguas abiertas.

---

## 3. Especies de Tortugas Marinas y Características de sus Rastros

Playa Mata Oscura se caracteriza por la superposición de ciclos reproductivos de cuatro especies amenazadas:

### Caracterización de la Anidación (Registros de la Fundación Agua y Tierra)

| Especie | Nombre Científico | Meses de Anidación | Nidos Aprox. / Temporada | Frecuencia Relativa |
| :--- | :--- | :---: | :---: | :---: |
| **Lora / Golfina** | *Lepidochelys olivacea* | Junio – Enero (Pico: Sep–Nov) | **> 100** | **Predominante** |
| **Verde** | *Chelonia mydas* | Diciembre – Febrero | ~30 | Frecuente |
| **Carey** | *Eretmochelys imbricata* | Junio – Agosto | ≤ 7 | Ocasional |
| **Baula / Canal** | *Dermochelys coriacea* | Enero – Febrero | ≤ 2 | Rara |

### Rasgos Distintivos de los Rastros en la Arena (Clave para Visión Artificial):
1. **Tortuga Baula (*D. coriacea*):** Rastro ancho y sumamente marcado (más de 1.5 m de ancho), dos líneas profundas generadas por sus extensas aletas delanteras, bordes difusos por el arrastre del plastrón y gran perturbación de arena en la zona del nido.
2. **Tortuga Verde (*C. mydas*):** Rastro moderado a ancho con dos huellas paralelas bien definidas, línea de arrastre continua y clara por el peso del animal, y excavación amplia de arena removida.
3. **Tortuga Carey (*E. imbricata*):** Rastro más estrecho y fino, marcas alternadas de aletas delanteras con patrón asimétrico e irregular, trayectoria curva y surco poco profundo.
4. **Tortuga Lora (*L. olivacea*):** Rastro pequeño y compacto, huellas poco profundas y muy juntas, trayectoria sinuosa, con menor contraste visual sobre arena seca.

---

## 4. ¿Por qué UAVs (Drones)? Justificación de la Perspectiva Aérea

El sistema prioriza el procesamiento de video capturado por **Vehículos Aéreos No Tripulados (UAV)** frente a métodos terrestres:

```
        Vuelo Cenital UAV (90° Perpendicular)           Patrullaje a Pie en Suelo
        ─────────────────────────────────────           ─────────────────────────
                  [ Dron UAV ] ✈                           🚶 Voluntario en suelo
                        │                                          │
                  FOV Amplio (15-25m)                      Línea visual bloqueada
                 ┌──────┴──────┐                           por dunas, troncos u olas.
                 ▼             ▼                                   │
            Arena mojada   Arena seca                      Riesgo de pisar nidos
            [Rastro A]     [Rastro B]                      y fatiga física severa.
```

1. **Amplitud del Campo Visual (FOV):** Un dron volando a una altitud de 15 a 25 metros barre una franja costera de 30 a 50 metros de ancho, permitiendo inspeccionar los 5 km de playa en menos de 20 minutos.
2. **Perspectiva Cenital Ortogonal (90°):** La vista perpendicular hacia abajo elimina la distorsión trapezoidal y captura la simetría completa de los surcos dejados por las aletas.
3. **Monitoreo No Destructivo:** No altera el hábitat costero, no aplasta nidos superficiales ni genera huellas de neumáticos que desorienten a los neonatos recién nacidos.
4. **Hardware UAV Utilizado en el Estudio:** Dron multirrotor **DJI Mavic 2 Enterprise** con cámara RGB integrada a baja altitud.
5. **Flexibilidad:** Aunque puede procesar cualquier video convencional, los filtros y pesos del modelo están optimizados para la cinemática de vuelo de un dron a velocidades de crucero de **3.0 a 5.0 m/s** (11 a 18 km/h).

---

## 5. Física del Sustrato: Arena Oscura (Mata Oscura) vs Arenas Claras (Europa/Mediterráneo)

Una de las contribuciones científicas más relevantes de este proyecto es haber resuelto la detección sobre **arena oscura**, un sustrato de altísima dificultad óptica:

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│        ARENA VOLCÁNICA OSCURA (PANAMÁ)       │        ARENAS CLARAS (EUROPA / MEDITERRÁNEO) │
├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
│ 🌋 Minerales ferromagnesianos oscuros.       │ 🏖️ Cuarzo y restos calcáreos de conchas.    │
│ 🌑 Coeficiente de reflectancia óptica bajísimo│ 🌕 Alta reflectancia óptica (brillo natural). │
│ 📉 Absorbe la luz lunar: casi nulo contraste │ 📈 La luz de la luna resalta las sombras del │
│    entre el surco del rastro y la superficie.│    surco con alto contraste fotométrico.     │
│ 🎯 Modelo debe aprender micro-texturas.      │ 👁️ Fácil detección con descriptores simples. │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

- **El Desafío de Mata Oscura:** Al ser una playa de arena oscura volcánica, la luz lunar nocturna no se refleja adecuadamente. Los modelos preentrenados en Europa o Medio Oriente fallan estrepitosamente porque asumen arena clara con alto contraste de sombras.
- **Garantía de Generalización:** Al lograr que **Turtle IA v3 alcance un 93% de precisión sobre arena oscura**, se garantiza científicamente que el modelo operará con un rendimiento igual o superior en cualquier playa de arena clara del mundo.

---

## 6. Bases de Datos y Evolución Histórica de los Datasets

Para superar los problemas de dominio, el proyecto evolucionó a lo largo de cuatro fases de datos:

```
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                      EVOLUCIÓN EXPERIMENTAL DE LOS MODELOS (DATASETS)                  │
 ├────────────────────────────────────────────────────────────────────────────────────────┤
 │                                                                                        │
 │  Fase 1: Base Roboflow (1,955 imgs)   ──►  Confundía hojas, ramas y sombras con tracks │
 │  Fase 2: ASGF Segmentación (ML + Nidos)──►  Inconsistente y muy lento en CPU (11.5 FPS)│
 │  Fase 3: Turtle v2 (Base + Hojas)     ──►  Sobre-penalizaba arena y omitía rastros     │
 │  Fase 4: Turtle IA v3 (2,515 imgs)    ──►  🎯 VERSIÓN FINAL DEFINITIVA (25.4 FPS CPU) │
 │                                                                                        │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### Detalle de las Cuatro Fases Experimentales:

1. **Fase 1: Dataset Base Internacional — Turtle Track Detector 2 (1,955 imágenes):**
   - Publicado por la *German University of Technology* ([Dataset Base en Roboflow](https://universe.roboflow.com/german-university-of-technology-f4tuz/turtle-track-detector-2)).
   - Clases: `Fox`, `Track`, `Turtle`.
   - *Fallo:* Confundía continuamente restos de vegetación costera (hojas secas de almendro de playa y palmeras) con rastros.
2. **Fase 2: Modelo ASGF — Segmentación Poligonal de Nidos con Machine Learning:**
   - Modelo `yolo26n-seg` con fotos de Playa Mata Oscura.
   - *Fallo:* Generación pesada de máscaras píxel a píxel (160×160) con latencia inaceptable de **86.88 ms (11.5 FPS)** en CPU.
3. **Fase 3: Versión Previa Experimental — Turtle v2:**
   - Incorporación de clase `Hojas` en Roboflow.
   - *Fallo inverso:* Al penalizar las sombras de la arena, **dejó de detectar rastros legítimos de tortugas (falsos negativos críticos)**.
4. **Fase 4: VERSIÓN FINAL DEFINITIVA Y GANADORA — Turtle IA v3 (2,515 imágenes):**
   - **Dataset Oficial Ganador:** Publicado en Roboflow Universe:
     👉 **[turtle-track-detector-2-205gp en Roboflow](https://universe.roboflow.com/snowcliks-workspace/turtle-track-detector-2-205gp)**
   - **Composición del Corpus:**
     - 1,955 imágenes del dataset base internacional.
     - ~500 imágenes reales de rastros y arena capturadas en **Playa Mata Oscura**.
     - ~100 imágenes negativas de vegetación costera y hojas de palma para suprimir falsos positivos.
     - Pipeline sintético **Adaptive Synthetic Few-Shot Pipeline (ASFP)** con rotaciones (±15°), variaciones de brillo/contraste y recortes.
   - **Clases (4):** `Fox` (depredador), `Hojas` (control de vegetación), `Track` (rastro de aletas), `Turtle` (tortuga).
   - **Desempeño:** **39.34 ms (25.4 FPS) en CPU** y desviación mínima **σ = 3.90 ms**.

---

## 7. ¿Por qué NO Machine Learning Tradicional vs Deep Learning?

| Criterio | Machine Learning Tradicional (Filtros + SVM/RF) | Deep Learning / Redes Neuronales (YOLO26-n) |
| :--- | :--- | :--- |
| **Comportamiento en Arena** | ❌ **Fracaso.** La arena no tiene geometría fija. La textura varía con la marea, grano y viento. | ✅ **Generalización Óptima.** Convoluciones jerárquicas aprenden patrones invariantes a escala y grano. |
| **Condiciones de Luz** | ❌ Umbrales rígidos fallan con sol cenital, sombras de nubes o penumbra nocturna. | ✅ Invarianza a luminosidad gracias a Batch Normalization y data augmentation. |
| **Carga en Inferencia** | ⚠️ Extracción manual de características densas en 640px es sumamente lenta en CPU. | ✅ Forward pass en arquitectura Nano optimizada en ONNX tarda menos de 40 ms. |
| **Falsos Positivos** | ❌ Imposible diferenciar una hoja seca de un surco de aleta mediante gradientes simples. | ✅ Discriminación semántica de alta fidelidad entre rastro biológico y hojarasca. |

---

## 8. ¿Qué es YOLO26 y por qué la Variante YOLO26-n? (ProgLoss, STAL y NMS-Free)

### ¿Qué es YOLO?
**YOLO (*You Only Look Once*)** es un detector en una sola etapa (*Single-Shot Detector*) que evalúa la imagen entera en una única pasada de red neuronal profunda, prediciendo simultáneamente bounding boxes y probabilidades de clase.

```
       Imagen Aérea de Dron (640×640)
                     │
                     ▼
      ┌─────────────────────────────┐
      │          YOLO26-n           │
      │   Backbone + Neck + Head    │
      │     (ProgLoss + STAL)       │
      └──────────────┬──────────────┘
                     │  (Forward Pass Directo End-to-End)
                     ▼
        Tensor Plano [1, 300, 6]  ──►  (x1, y1, x2, y2, confianza, clase)
```

### Mecanismos Avanzados de YOLO26:
1. **ProgLoss (Progressive Loss):** Función de pérdida progresiva que balancea la localización de objetos pequeños frente a objetos grandes durante el entrenamiento, vital para identificar los surcos milimétricos de las aletas.
2. **STAL (Spatio-Temporal Alignment Layer):** Capa de alineamiento espacio-temporal que afina las características visuales en imágenes aéreas de bajo contraste.
3. **Esquema NMS-Free (End-to-End):** A diferencia de detectores convencionales que requieren el algoritmo iterativo Non-Maximum Suppression (NMS) en CPU, YOLO26 predice directamente un conjunto disjunto de detecciones `[1, 300, 6]`. Esto **elimina la latencia variable** dependiente del número de objetos y otorga una velocidad constante y predecible.
4. **Variante Nano (`yolo26n`):** Con solo **2.4 millones de parámetros**, minimiza el uso de memoria caché y permite una ejecución fluida en CPU.

---

## 9. ¿Por qué ONNX Runtime y NO Inferencia Directa con Python?

```
 [ENTORNO DE PRODUCCIÓN EN LA PLATAFORMA]
  Video Frame (Web / Canvas)
          │
          ▼
  Tauri Webview (Wasm SIMD Threaded) / Rust IPC
          │
          ▼
  ONNX Runtime (CPUExecutionProvider) ──► Fusión de Capas (Conv+BN), Vectorización AVX2
          │
          ▼
  Detección en 39.34 ms (25.4 FPS) ── (Sin Python, Sin CUDA, Sin PyTorch)
```

- **Sin Dependencias Pesadas:** Elimina la necesidad de instalar Python (2.5 GB+ de PyTorch/CUDA) en las laptops de campo de la fundación.
- **Sin Bloqueo del GIL:** Ejecución nativa sin el bloqueo del intérprete de Python.
- **Optimizaciones de Grafo:** Fusión de capas (`Conv2d + BatchNorm + SiLU`) y vectorización mediante instrucciones de CPU (**AVX2, AVX-512, SSE**).
- **Aceleración Empírica:** **1.49× a 1.70× más rápido que PyTorch CPU** (ahorro de hasta 47 ms por fotograma).

---

## 10. Arquitectura de Software: ¿Por qué Tauri (Rust + React + Vite + Tailwind)?

| Parámetro | Electron (VS Code / Slack) | Python GUI (PyQt / Tkinter) | **Tauri v2 (Solución Elegida)** |
| :--- | :---: | :---: | :---: |
| **Peso del Instalador** | 150 MB – 300 MB | 80 MB – 180 MB | **< 15 MB – 25 MB** |
| **Consumo de Memoria RAM** | 400 MB – 800 MB | 150 MB – 300 MB | **< 40 MB – 70 MB** |
| **Motor Web** | Incrusta Chromium entero | No aplica / Obsoleto | **Usa WebView nativo del SO (WebView2)** |
| **Rendimiento Backend** | Node.js (Monohilo, pesado) | Lento, interpretado, bloqueo GIL | **Rust: Memory-safety, concurrencia nativa sin GC** |
| **Multiplataforma** | Windows, Linux, macOS | Empaquetado frágil | **Windows 11/10, Linux, macOS, iOS, Android y Web** |

- **Gestor de Paquetes:** Uso estricto de **pnpm** para instalaciones deterministas y eficientes en espacio.
- **Frontend Moderno:** React 18, TypeScript, Tailwind CSS y Zustand para control de estado sin sobrecarga.

---

## 11. Entrenamiento del Modelo: Local vs Google Colab

El script oficial parametrizado es [`Plataforma/training/colab_train.py`](file:///c:/Users/tavog/Documents/Codigo/Universidad/Tortugasv2/Plataforma/training/colab_train.py).

```
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│         ENTRENAMIENTO LOCAL          │       │      ENTRENAMIENTO GOOGLE COLAB      │
├──────────────────────────────────────┤       ├──────────────────────────────────────┤
│ ⏱ Tiempo: 12 a 16 Horas (CPU/GPU mod)│       │ ⏱ Tiempo: 45 a 90 Minutos            │
│ 🔥 Temperatura: Alto estrés térmico  │       │ ☁️ Hardware: NVIDIA T4 / A100 GPU    │
│ 💻 Recursos: Computadora saturada    │       │ 🔋 Cero consumo de batería local     │
│ ⚠️ Riesgo: Interrupciones en la PC   │       │ 🚀 Exportación automática a ONNX     │
└──────────────────────────────────────┘       └──────────────────────────────────────┘
```

- **Entrenamiento Local:** En CPU tarda **12 a 16 horas**, sobrecalienta la laptop y satura el procesador.
- **Entrenamiento en Google Colab:** Con GPU T4 toma menos de **1 hora**, con Mixed Precision (FP16), escalado lineal de `lr0`, Early Stopping y exportación automática a `.onnx`.

---

## 12. Validación Científica de Hipótesis y Resultados Estadísticos

### Hipótesis Científica Evaluada:
> *"El modelo propuesto YOLO26-n ejecutado sobre el motor optimizado ONNX Runtime CPU presenta una latencia de inferencia significativamente menor que las arquitecturas nano competidoras (YOLOv5-nu, YOLOv8-n y YOLO11-n), superando el umbral operativo crítico de 20 FPS en un procesador convencional sin tarjeta gráfica."*

- **Población Muestral:** **1,100 inferencias cronometradas** (100 por modelo tras 15 iteraciones de calentamiento *Warm-up*).
- **Entorno Experimental:** Laptop Victus by HP (AMD Ryzen 5 @ 4.30 GHz) / Intel Core i5 11.ª Gen, Windows 11, Batch = 1, entrada 640×640 px.

---

### 12.1 Tabla General de Desempeño Experimental (N = 100 mediciones por modelo)

| Arquitectura y Configuración | Motor | Tarea | Media (μ) [ms] | Desv. Est. (σ) [ms] | Mediana [ms] | P95 [ms] | IC 95% [ms] | Rendimiento (FPS) | Estado de Tiempo Real |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🟢 **YOLO26-n (Tortugas v3 - Final)** | **ONNX** | **Detección** | **39.34** | **3.90** | **39.55** | **44.23** | **[38.57, 40.12]** | **25.4 FPS** | ✅ **Cumple (Óptimo)** |
| 🟢 **YOLO26-n (Turtle v2 - Previa)** | **ONNX** | **Detección** | **41.11** | 9.79 | **39.94** | **45.99** | [39.17, 43.05] | **24.3 FPS** | ✅ **Cumple** |
| 🟡 **YOLO26-n (Extranjero Base)** | **ONNX** | **Detección** | 51.27 | 16.29 | 50.04 | 62.10 | [48.04, 54.50] | 19.5 FPS | ⚠️ Al límite (50 ms) |
| 🔵 **YOLO11-n** | **ONNX** | Detección | 58.68 | 12.34 | 56.90 | 76.88 | [56.23, 61.13] | 17.0 FPS | ❌ No alcanza 20 FPS |
| 🔵 **YOLOv5-nu** | **ONNX** | Detección | 64.76 | 8.38 | 63.90 | 73.49 | [63.10, 66.42] | 15.4 FPS | ❌ No alcanza 20 FPS |
| 🔵 **YOLOv8-n** | **ONNX** | Detección | 71.93 | 9.52 | 71.74 | 80.34 | [70.05, 73.82] | 13.9 FPS | ❌ No alcanza 20 FPS |
| 🟣 **YOLO26-n-seg (ASGF Nidos)** | **ONNX** | Segmentación | 86.88 | 11.71 | 86.51 | 96.69 | [84.56, 89.20] | 11.5 FPS | ❌ Penalización Máscaras |
| 🔴 **YOLO11-n [PyTorch]** | PyTorch | Detección | 99.86 | 18.92 | 95.41 | 109.70 | [96.11, 103.62] | 10.0 FPS | ❌ Inferencia Nativa Lenta |
| 🔴 **YOLOv5-nu [PyTorch]** | PyTorch | Detección | 101.75 | 23.35 | 93.71 | 137.75 | [97.12, 106.39] | 9.8 FPS | ❌ Inferencia Nativa Lenta |
| 🔴 **YOLOv8-n [PyTorch]** | PyTorch | Detección | 107.27 | 26.78 | 100.39 | 154.86 | [101.95, 112.58] | 9.3 FPS | ❌ Inferencia Nativa Lenta |
| 🔴 **YOLO26-n-seg [PyTorch]** | PyTorch | Segmentación | 133.97 | 22.76 | 130.43 | 165.81 | [129.45, 138.48] | 7.5 FPS | ❌ Inferencia Nativa Lenta |

---

### 12.2 Comparativa Visual de las Variantes del Proyecto

![Figura 5: Comparativa de Latencia y FPS entre Modelos del Proyecto](./figure5_project_onnx_models_comparison.png)

*Figura 5: (Izquierda) Diagrama de cajas de la latencia en CPU de las 4 variantes del proyecto frente al umbral crítico de 50 ms. (Derecha) Rendimiento en FPS frente al estándar de 20 FPS para video continuo de dron.*

---

### 12.3 Pruebas de Normalidad, ANOVA y Comparaciones Tukey HSD

- **Normalidad de Shapiro-Wilk:** `YOLO26-n (Tortugas v3)` presentó distribución normal (**W = 0.9827, p = 0.2154 > 0.05**), reflejando estabilidad absoluta.
- **Homocedasticidad de Levene:** W = 6.8618, p < 0.001 (heterogeneidad de varianzas, justificando el doble análisis paramétrico y no paramétrico).
- **ANOVA de Un Factor:** **F = 343.66, p < 10⁻¹⁰⁰** (Rechazo contundente de H₀).
- **Kruskal-Wallis:** **H = 998.41, df = 10, p < 10⁻²⁰⁰** (Rechazo contundente de H₀).
- **Tukey HSD Post-hoc (YOLO26-n v3 frente a rivales):**
  - vs **YOLO11-n [ONNX]**: **-19.34 ms (+33.0% más rápido)**, `p < 10⁻¹⁴` (Significativo).
  - vs **YOLOv5-nu [ONNX]**: **-25.42 ms (+39.3% más rápido)**, `p < 10⁻¹⁵` (Significativo).
  - vs **YOLOv8-n [ONNX]**: **-32.59 ms (+45.3% más rápido)**, `p < 10⁻¹⁵` (Significativo).
  - vs **Modelo Extranjero Base**: **-11.93 ms (+23.3% más rápido)**, `p = 1.79 × 10⁻⁵` (Significativo).

![Figura 3: Intervalos de Confianza al 95% de la Latencia Media en CPU](./figure3_tukey_hsd_intervals.png)

*Figura 3: Intervalos de confianza al 95% (Forest Plot). Los intervalos de YOLO26-n (franja de 38.5 a 40.1 ms) están completamente disjuntos de los competidores.*

![Figura 2: Impacto de la Optimización de Runtime](./figure2_onnx_vs_pytorch_speedup.png)

*Figura 2: Factor multiplicativo de aceleración (Speedup) alcanzado mediante ONNX Runtime frente a PyTorch CPU (1.49× a 1.70×).*

---

## 13. Desempeño según Iluminación y Matriz Exhaustiva de Limitaciones

### Desempeño Experimental según Iluminación en Campo:
- **Luz Natural Diurna / Matutina (5:00 a.m. – 6:50 p.m.):** **87% a 93% de precisión**. La luz rasante del amanecer y atardecer acentúa el relieve tridimensional de los surcos de las aletas.
- **Horario Nocturno (7:00 p.m. – 12:00 a.m., horario real de patrullajes a pie):** **61% a 72% de precisión**. La escasa reflectancia lunar sobre arena volcánica oscura reduce el contraste visual en cámaras RGB estándar. Representa el principal reto pendiente, motivando la incorporación de iluminación complementaria en el UAV o sensores térmicos infrarrojos.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        MATRIZ DE CAPACIDADES Y LIMITACIONES                            │
├──────────────────────────────────────────────────┬─────────────────────────────────────┤
│                 QUÉ ES POSIBLE                   │          QUÉ NO ES POSIBLE          │
├──────────────────────────────────────────────────┼─────────────────────────────────────┤
│ ✅ Detección en tiempo real en CPU (25.4 FPS).   │ ❌ Conteo de huevos dentro del nido │
│ ✅ Inferencia 100% offline sin internet en playa.│    (requiere inspección física).    │
│ ✅ Discriminación de vegetación/hojas y zorros.  │ ❌ Visión nocturna en oscuridad sin │
│ ✅ Seguimiento de trayectoria (Tracking ID).     │    luz auxiliar o cámara térmica.   │
│ ✅ Exportación tabular inmediata (CSV y JSON).   │ ❌ Vuelo con vientos > 30 km/h o    │
│ ✅ Identificación visual en resoluciones 640px.  │    lluvia tropical intensa.         │
│                                                  │ ❌ Estimación métrica de profundidad│
│                                                  │    del nido sin sensor LiDAR.       │
└──────────────────────────────────────────────────┴─────────────────────────────────────┘
```

### Limitaciones Físicas y Operativas:
1. **Altitud de Vuelo Estricta:** Óptima entre **15 y 25 metros**. Volar a más de 30 metros difumina la firma del rastro; volar a menos de 10 metros perturba a las aves y reduce el área de barrido.
2. **Dinámica de Mareas:** El oleaje de marea alta borra los rastros intermareales. Se debe coordinar el vuelo durante la bajamar matutina.
3. **Fuerza del Viento:** Vientos sostenidos mayores a 25-30 km/h desestabilizan el dron y borran la cresta de la arena seca.

---

## 14. Estudio Social y Percepción Comunitaria (Alineación ODS)

El proyecto integró una encuesta de percepción social a **43 participantes** (36 de la cabecera de Santiago y 7 de la comunidad costera de Mariato):
- **88.4% de Apoyo Comunitario:** Fuerte respaldo social a las iniciativas de conservación ambiental asistidas por tecnología.
- **87.8% de Aceptación Tecnológica:** Reconocimiento explícito del dron y la visión por computador como herramientas legítimas y no invasivas para apoyar el patrullaje de los guardaparques.
- **Alineación con los ODS de la ONU:**
  - **ODS 14 (Vida Submarina):** Protección de especies marinas en peligro crítico.
  - **ODS 15 (Ecosistemas Terrestres y Costeros):** Preservación de la franja intermareal de anidación.
  - **ODS 17 (Alianzas para lograr los Objetivos):** Cooperación sinérgica entre la academia (Universidad Tecnológica de Panamá), organizaciones comunitarias (Fundación Agua y Tierra) y la comunidad local.

---

## 15. Requisitos de Hardware y Software

| Componente | Requisito Mínimo (Laptop de Campo) | Requisito Recomendado (Rendimiento Óptimo) |
| :--- | :--- | :--- |
| **Procesador (CPU)** | Intel Core i3 / AMD Ryzen 3 (4 hilos, soporte AVX2) | Intel Core i5 / i7 (11.ª gen+) / AMD Ryzen 5 / 7 |
| **Memoria RAM** | 8.0 GB DDR4 | 16.0 GB DDR4 / DDR5 |
| **Tarjeta Gráfica (GPU)**| **No requerida** (Inferencia 100% sobre CPU) | GPU dedicada opcional para aceleración adicional |
| **Almacenamiento** | 2.0 GB de espacio libre (SSD recomendado) | SSD NVMe M.2 |
| **Conectividad** | **100% Offline** (No requiere internet en campo) | Conexión para sincronización inicial de paquetes |
| **Sistema Operativo** | Windows 10/11 (64-bit), Linux (Ubuntu 22.04+), macOS | Windows 11 (64-bit, Build 22621 o superior) |

---

## 16. Guía Paso a Paso para Reproducir el Proyecto

### Paso 1: Clonar el Repositorio Oficial
```bash
git clone https://github.com/Snowclik/sea-turtle-track.git
cd sea-turtle-track
```

### Paso 2: Instalar Dependencias con pnpm
```bash
cd Plataforma/apps/desktop
pnpm install
```

### Paso 3: Ejecutar la Aplicación en Modo Desarrollo
```bash
pnpm run tauri dev
```

### Paso 4: Compilar el Instalador para Windows (.exe / .msi)
```bash
pnpm run tauri build
```

### Paso 5: Reproducir los Ensayos Estadísticos (Opcional)
```bash
cd ../../../
pip install -r requirements.txt
python hypothesis_cpu_benchmark.py
```

### Paso 6: Reentrenamiento en Google Colab (Opcional)
1. Abrir [`Plataforma/training/colab_train.py`](file:///c:/Users/tavog/Documents/Codigo/Universidad/Tortugasv2/Plataforma/training/colab_train.py) en Google Colab con GPU T4.
2. Descargar el dataset oficial ganador: [Roboflow Universe - turtle-track-detector-2-205gp](https://universe.roboflow.com/snowcliks-workspace/turtle-track-detector-2-205gp).
3. Ejecutar el pipeline de entrenamiento y exportación a ONNX.

---

## 17. Estructura del Repositorio

```
sea-turtle-track/
├── Plataforma/                           # Ecosistema de la aplicación de usuario
│   ├── apps/
│   │   └── desktop/                      # Plataforma de escritorio en Tauri + React
│   │       ├── public/
│   │       │   ├── models/               # Modelos ONNX compilados y metadatos JSON
│   │       │   │   ├── model_v3.onnx     # 🎯 Modelo oficial de producción (Turtle IA v3)
│   │       │   │   ├── model_v3_meta.json
│   │       │   │   ├── model_v2.onnx
│   │       │   │   ├── model_ext.onnx
│   │       │   │   └── model_asgf.onnx
│   │       │   ├── ort-wasm-simd-threaded.wasm
│   │       │   └── ort-wasm-simd-threaded.asyncify.wasm
│   │       ├── src/                      # Frontend en React 18 + TypeScript + Tailwind
│   │       │   ├── components/           # VideoPlayer, ControlBar, StatsPanel, ExportPanel
│   │       │   ├── hooks/                # useVideoAnalysis.ts, useImageAnalysis.ts
│   │       │   ├── services/             # onnxService.ts, tracker.ts (algoritmo de tracking)
│   │       │   ├── store/                # analysisStore.ts (Zustand state management)
│   │       │   └── types/                # Interfaces y contratos de tipos en TypeScript
│   │       ├── src-tauri/                # Backend nativo de alto rendimiento en Rust
│   │       │   ├── src/                  # Comandos IPC nativos y exportadores CSV/JSON
│   │       │   ├── Cargo.toml            # Manifiesto de dependencias de Rust
│   │       │   └── tauri.conf.json       # Configuración de ventanas, permisos y build de Tauri
│   │       ├── package.json              # Dependencias de JavaScript
│   │       └── pnpm-lock.yaml            # Bloqueo determinista de versiones con pnpm
│   ├── training/
│   │   └── colab_train.py                # Script integral para Google Colab con exportación ONNX
│   ├── SYSTEM_DESIGN.md                  # Documento de arquitectura maestra del sistema
│   └── readme.md                         # Resumen del sub-módulo de escritorio
├── configs/                              # Configuraciones YAML reproducibles
│   ├── augmentation.yaml                 # Parámetros de aumento sintético de datos
│   ├── synthetic_generation.yaml         # Configuración del motor ASFP
│   └── training_protocol.yaml            # Hiperparámetros de entrenamiento
├── src/                                  # Código fuente modular de experimentación Python
│   ├── synthetic_generation/             # Motor de generación sintética de imágenes
│   └── tools/                            # Comparadores y analizadores estructurales de modelos
├── hypothesis_cpu_benchmark.py           # Protocolo experimental de 1,100 inferencias cronometradas
├── informe_validacion_hipotesis_cpu.md   # Informe científico exhaustivo de la hipótesis evaluada
├── figure1_all_models_cpu_latency.png    # Distribución global de latencia (11 modelos)
├── figure2_onnx_vs_pytorch_speedup.png   # Gráfica de aceleración (Speedup X)
├── figure3_tukey_hsd_intervals.png       # Intervalos de confianza al 95% (Forest Plot)
├── figure5_project_onnx_models_comparison.png # Boxplot y FPS de los modelos del proyecto
├── CITATION.cff                          # Metadatos para el botón 'Cite this repository' en GitHub
├── LICENSE                               # Términos legales de código abierto con citación obligatoria
├── requirements.txt                      # Dependencias de Python para reproducir benchmarks
├── .gitignore                            # Protección estricta contra fugas de datos y binarios
└── README.md                             # Documentación principal del repositorio
```

---

## 18. Propiedad Intelectual, Delimitación de Autoría y Citación Obligatoria

Para salvaguardar la autoría científica y definir de manera inequívoca qué elementos son de creación propia del proyecto frente a las herramientas y bibliotecas de terceros, se establece la siguiente delimitación formal:

### 18.1 Delimitación de Autoría: ¿Qué creamos nosotros vs. Herramientas de Terceros?

```
┌─────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
│           COMPONENTES DE TERCEROS UTILIZADOS                │        PROPIEDAD INTELECTUAL Y CREACIÓN PROPIA DEL PROYECTO │
├─────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
│ • Arquitectura base de YOLO (Ultralytics y autores).        │ • El Sistema Integral de Monitoreo Aéreo para Drones.       │
│ • Plataforma y almacenamiento de Roboflow Inc.              │ • La Plataforma de Escritorio Tauri (Rust + React + Wasm).  │
│ • Motor de compilación Tauri (Fundación Tauri).             │ • El Pipeline Adaptativo de Aumento Sintético (ASFP).       │
│ • Librerías base (React, Vite, Tailwind, ONNX Runtime).     │ • El Dataset Curado de Mata Oscura (turtle-track-205gp).    │
│ • Dataset inicial internacional (GUtech en Roboflow).       │ • La Calibración y Pesos Oficiales de Turtle IA v3.         │
│                                                             │ • El Protocolo de Validación en CPU (1,100 inferencias).    │
└─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘
```

### 18.2 Términos de Licenciamiento y Código Abierto:
1. **Uso Libre para Aprendizaje y Pruebas Personales:**  
   Cualquier desarrollador, estudiante o investigador es libre de clonar, compilar y ejecutar este software en su computadora personal con propósitos didácticos o de inspección técnica.
2. **Citación y Atribución Obligatoria en Investigaciones o Productos:**  
   Si utilizas este software, sus modelos entrenados (`model_v3.onnx`), su metodología de data augmentation adaptativo, o sus datos en:
   - Artículos científicos, pósteres, congresos o conferencias académicas;
   - Tesis de licenciatura, maestría o doctorado;
   - Informes técnicos ambientales o proyectos institucionales;
   - Desarrollos de software o productos públicos derivados;  
   **ES UN REQUISITO OBLIGATORIO Y ESTRICTO CITAR ESTE REPOSITORIO Y EL PROYECTO** como fuente bibliográfica original.
3. **Cláusula Anti-Plagio Académico:**  
   Replicar o presentar la metodología, los experimentos, los datos o la arquitectura de este sistema como creación propia sin citar la fuente constituye una falta grave contra la ética científica y una infracción directa de los derechos de autor protegidos por la Ley de Derecho de Autor de Panamá, el Convenio de Berna y la OMPI. El equipo de investigación ejercerá las acciones institucionales pertinentes ante universidades, conferencias o comités de ética académica en caso de detectar apropiaciones no atribuidas.
4. **Restricción Comercial:**  
   El uso o comercialización lucrativa directa del sistema sin autorización escrita previa está estrictamente prohibido.

---

### 18.3 Cómo Citar este Trabajo (Formatos Oficiales)

GitHub detecta automáticamente el archivo [`CITATION.cff`](file:///c:/Users/tavog/Documents/Codigo/Universidad/Tortugasv2/CITATION.cff) en la raíz del repositorio, habilitando el botón **"Cite this repository"** en la barra lateral derecha. Puedes copiar la cita en los siguientes formatos:

#### Formato BibTeX:
```bibtex
@software{seaturtletrack2026,
  author = {{Sea Turtle Track Research Team}},
  title = {Sistema Inteligente Basado en UAV y YOLO26 para la Detección de Rastros de Anidación de Tortugas Marinas},
  institution = {Universidad Tecnológica de Panamá (UTP)},
  year = {2026},
  publisher = {GitHub},
  version = {3.0},
  url = {https://github.com/Snowclik/sea-turtle-track}
}
```

#### Formato APA (7.ª edición):
```text
Sea Turtle Track Research Team. (2026). Sistema Inteligente Basado en UAV y YOLO26 para la Detección de Rastros de Anidación de Tortugas Marinas (Versión 3.0) [Software y Modelos de Inferencia]. Universidad Tecnológica de Panamá (UTP). https://github.com/Snowclik/sea-turtle-track
```

#### Formato IEEE:
```text
[1] Sea Turtle Track Research Team, "Sistema Inteligente Basado en UAV y YOLO26 para la Detección de Rastros de Anidación de Tortugas Marinas," Universidad Tecnológica de Panamá (UTP), 2026. [Online]. Disponible: https://github.com/Snowclik/sea-turtle-track.
```

---

## 19. Agradecimientos y Licencia

### Reconocimientos Institucionales:
- **Universidad Tecnológica de Panamá (UTP):** Por el respaldo académico, institucional y la promoción de la investigación formativa mediante la **Jornada de Iniciación Científica (JIC)**.
- **Secretaría Nacional de Ciencia, Tecnología e Innovación (SENACYT):** Por fomentar la ciencia y la innovación en Panamá a través del financiamiento y organización de la **JIC Nacional** en el marco del **X Congreso IESTEC**.
- **Fundación Agua y Tierra (Playa Mata Oscura, Mariato, Veraguas):** Por su invaluable dedicación comunitaria a la conservación de las tortugas marinas, la facilitación del acceso al área de estudio y el acompañamiento biológico en campo.
- **German University of Technology in Oman (GUtech):** Por poner a disposición pública el dataset inicial `turtle-track-detector-2` en Roboflow Universe.

### Licencia Legal:
Este proyecto se rige por los términos establecidos en el archivo [`LICENSE`](LICENSE), que combina el modelo de código abierto para fines educativos y de conservación con una **cláusula obligatoria de citación científica, atribución de autoría y prohibición de plagio**.

<div align="center">

---

**Sea Turtle Track** — *Ciencia, Tecnología e Innovación al Servicio de la Conservación de los Océanos.*

</div>

