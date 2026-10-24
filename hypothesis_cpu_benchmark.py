#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=================================================================================================
PROYECTO: SISTEMA INTELIGENTE BASADO EN UAV Y YOLO26 PARA LA DETECCIÓN DE RASTROS DE TORTUGAS
=================================================================================================
Validación Experimental Exhaustiva de la Hipótesis Científica:
    "Comparación de la Velocidad de Inferencia (CPU)"

Objetivo:
    Demostrar empíricamente y con rigor bioestadístico que YOLO26-n presenta una latencia
    de inferencia en CPU significativamente menor que sus modelos predecesores (YOLOv5-n, 
    YOLOv8-n, YOLO11-n), garantizando monitoreo en tiempo real (>20 FPS) en procesadores 
    embebidos/locales sin requerir GPUs dedicadas de alto consumo para vehículos aéreos no tripulados.

Alcance de Modelos Evaluados:
    1. Modelos ONNX Runtime CPU (Motor oficial de la plataforma de escritorio):
       - YOLO26-n (Turtle v2 - Detección Oficial)
       - YOLO26-n (Tortugas v3 - Detección Optimizada)
       - YOLO26-n (Ext Base - Detección Base)
       - YOLO26-n-seg (Nidos ASGF - Segmentación de Instancias)
       - YOLO11-n (Competidor Oficial Baseline ONNX)
       - YOLOv8-n (Competidor Oficial Baseline ONNX)
       - YOLOv5-nu (Competidor Oficial Baseline ONNX)

    2. Modelos Nativos PyTorch CPU (Framework de entrenamiento y desarrollo):
       - YOLO26-n-seg (PyTorch .pt)
       - YOLO11-n (PyTorch .pt)
       - YOLOv8-n (PyTorch .pt)
       - YOLOv5-nu (PyTorch .pt)

Protocolo Estadístico:
    - Muestreo controlado con calentamiento térmico (warm-up).
    - Estadísticos descriptivos robustos (Media, DE, Mediana, IQR, IC 95%, Percentiles P95/P99, FPS).
    - Contrastes de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov) y homocedasticidad (Levene).
    - Pruebas paramétricas (One-Way ANOVA + Tukey HSD post-hoc con corrección de Tukey).
    - Pruebas no paramétricas (Kruskal-Wallis + Mann-Whitney U con corrección de Bonferroni).
    - Análisis comparativo de aceleración (Speedup ONNX Runtime vs PyTorch CPU).
=================================================================================================
"""

import os
import sys
import time
import shutil
import platform
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Ultralytics & PyTorch
try:
    from ultralytics import YOLO
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ Error: PyTorch y Ultralytics son necesarios.")
    sys.exit(1)

# ONNX Runtime
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("❌ Error: ONNX Runtime es requerido.")
    sys.exit(1)


# ===============================================================================================
# CONFIGURACIÓN EXPERIMENTAL
# ===============================================================================================
CONFIG = {
    "num_samples": 100,        # 100 mediciones cronometradas por modelo (1,100 inferencias totales)
    "warmup_runs": 15,         # 15 iteraciones de calentamiento para estabilización de caché
    "img_size": (640, 640),    # Resolución estándar 640x640 px
    "batch_size": 1,           # Batch = 1 (inferencia secuencial en tiempo real en dron UAV)
    "alpha": 0.05,             # Nivel de significancia alfa = 0.05 (95% nivel de confianza)
    "output_dir": "reports/hypothesis_latency_cpu",
    "downloads_mirror": "C:/Users/tavog/Downloads/hypothesis_latency_cpu",
    "random_seed": 42
}

np.random.seed(CONFIG["random_seed"])
torch.manual_seed(CONFIG["random_seed"])

# ===============================================================================================
# CATÁLOGO DE MODELOS EXPERIMENTALES (11 MODELOS / CONFIGURACIONES)
# ===============================================================================================
MODEL_EXPERIMENTS = [
    # ── 1. Modelos ONNX Runtime CPU (Motor de Producción) ──
    {
        "id": "YOLO26-n (Turtle v2) [ONNX]",
        "engine": "onnx",
        "family": "YOLO26",
        "task": "Detección",
        "path": "Plataforma/apps/desktop/public/models/model_v2.onnx",
        "is_project": True,
        "is_target": True,
        "color": "#2ca02c"
    },
    {
        "id": "YOLO26-n (Tortugas v3) [ONNX]",
        "engine": "onnx",
        "family": "YOLO26",
        "task": "Detección",
        "path": "Plataforma/apps/desktop/public/models/model_v3.onnx",
        "is_project": True,
        "is_target": False,
        "color": "#3cb44b"
    },
    {
        "id": "YOLO26-n (Ext Base) [ONNX]",
        "engine": "onnx",
        "family": "YOLO26",
        "task": "Detección",
        "path": "Plataforma/apps/desktop/public/models/model_ext.onnx",
        "is_project": True,
        "is_target": False,
        "color": "#8cd17d"
    },
    {
        "id": "YOLO26-n-seg (Nidos) [ONNX]",
        "engine": "onnx",
        "family": "YOLO26",
        "task": "Segmentación",
        "path": "Plataforma/apps/desktop/public/models/model_asgf.onnx",
        "is_project": True,
        "is_target": False,
        "color": "#17becf"
    },
    {
        "id": "YOLO11-n [ONNX]",
        "engine": "onnx",
        "family": "YOLO11",
        "task": "Detección",
        "path": "yolo11n.onnx",
        "is_project": False,
        "is_target": False,
        "color": "#1f77b4"
    },
    {
        "id": "YOLOv8-n [ONNX]",
        "engine": "onnx",
        "family": "YOLOv8",
        "task": "Detección",
        "path": "yolov8n.onnx",
        "is_project": False,
        "is_target": False,
        "color": "#ff7f0e"
    },
    {
        "id": "YOLOv5-nu [ONNX]",
        "engine": "onnx",
        "family": "YOLOv5",
        "task": "Detección",
        "path": "yolov5nu.onnx",
        "is_project": False,
        "is_target": False,
        "color": "#9467bd"
    },

    # ── 2. Modelos PyTorch CPU (Framework de Desarrollo Nativo) ──
    {
        "id": "YOLO26-n-seg [PyTorch]",
        "engine": "pytorch",
        "family": "YOLO26",
        "task": "Segmentación",
        "path": "yolo26n-seg.pt",
        "is_project": True,
        "is_target": False,
        "color": "#4363d8"
    },
    {
        "id": "YOLO11-n [PyTorch]",
        "engine": "pytorch",
        "family": "YOLO11",
        "task": "Detección",
        "path": "yolo11n.pt",
        "is_project": False,
        "is_target": False,
        "color": "#469990"
    },
    {
        "id": "YOLOv8-n [PyTorch]",
        "engine": "pytorch",
        "family": "YOLOv8",
        "task": "Detección",
        "path": "yolov8n.pt",
        "is_project": False,
        "is_target": False,
        "color": "#e6194b"
    },
    {
        "id": "YOLOv5-nu [PyTorch]",
        "engine": "pytorch",
        "family": "YOLOv5",
        "task": "Detección",
        "path": "yolov5nu.pt",
        "is_project": False,
        "is_target": False,
        "color": "#f032e6"
    },
]


# ===============================================================================================
# 1. ESPECIFICACIONES DE HARDWARE Y ENTORNO
# ===============================================================================================
def log_hardware_environment() -> Dict[str, Any]:
    """Registra y valida las características técnicas del procesador host."""
    print("\n" + "="*85)
    print("🔬 ENTORNO DE EJECUCIÓN Y ESPECIFICACIONES DE HARDWARE (CPU)")
    print("="*85)
    
    cpu_info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "pytorch_cpu_threads": torch.get_num_threads(),
    }
    
    print(f"• Sistema Operativo     : {cpu_info['os']} {cpu_info['os_release']}")
    print(f"• Procesador CPU        : {cpu_info['processor'] or 'Intel/AMD x86_64 CPU'}")
    print(f"• Cores Físicos / Lóg.  : {cpu_info['cpu_physical_cores']} físicos | {cpu_info['cpu_logical_cores']} hilos lógicos")
    print(f"• Memoria RAM Total     : {cpu_info['ram_gb']} GB")
    print(f"• Versión PyTorch       : {cpu_info['torch_version']} (Threads: {cpu_info['pytorch_cpu_threads']})")
    print(f"• Versión ONNX Runtime  : {cpu_info['onnxruntime_version']} (Provider: CPUExecutionProvider)")
    print(f"• Hardware de Prueba    : CPU host local para emulación directa de hardware embebido UAV")
    print("="*85 + "\n")
    return cpu_info


# ===============================================================================================
# 2. BENCHMARK EN ONNX RUNTIME CPU
# ===============================================================================================
def run_benchmark_onnx(
    model_name: str,
    onnx_path: str,
    num_samples: int,
    warmup_runs: int,
    img_size: Tuple[int, int]
) -> List[float]:
    """Ejecuta inferencia directa con ONNX Runtime optimizado para CPU."""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = min(8, psutil.cpu_count(logical=False) or 4)
    
    session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    
    # Tensor de entrada normalizado [1, 3, H, W] float32
    dummy_input = np.random.uniform(0.0, 1.0, size=(1, 3, img_size[0], img_size[1])).astype(np.float32)
    
    # Calentamiento
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_input})
        
    latencies: List[float] = []
    for _ in range(num_samples):
        frame = np.random.uniform(0.0, 1.0, size=(1, 3, img_size[0], img_size[1])).astype(np.float32)
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: frame})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        
    return latencies


# ===============================================================================================
# 3. BENCHMARK EN PYTORCH CPU
# ===============================================================================================
def run_benchmark_pytorch(
    model_name: str,
    weight_path: str,
    num_samples: int,
    warmup_runs: int,
    img_size: Tuple[int, int]
) -> List[float]:
    """Ejecuta inferencia directa con PyTorch en CPU."""
    model = YOLO(weight_path)
    if hasattr(model, 'model') and model.model is not None:
        model.model.to("cpu")
        model.model.eval()
        
    dummy = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float32)
    
    # Calentamiento
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy, verbose=False)
            
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(num_samples):
            frame = torch.randn(1, 3, img_size[0], img_size[1], dtype=torch.float32)
            t0 = time.perf_counter()
            _ = model(frame, verbose=False)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
    return latencies


# ===============================================================================================
# 4. ANÁLISIS ESTADÍSTICO DE HIPÓTESIS
# ===============================================================================================
def analyze_statistics(df: pd.DataFrame, alpha: float = 0.05) -> Tuple[Dict[str, Any], Any]:
    """Calcula todas las métricas descriptivas e inferenciales."""
    models = list(df["model"].unique())
    groups = [df[df["model"] == m]["latency_ms"].values for m in models]
    
    results: Dict[str, Any] = {
        "alpha": alpha,
        "models": models,
        "descriptive": {},
        "normality": {},
        "homoscedasticity": {},
        "anova": {},
        "tukey_hsd": [],
        "kruskal_wallis": {},
        "mann_whitney_bonferroni": [],
        "speedup_onnx_vs_pytorch": {},
        "hypothesis_decision": {}
    }
    
    # 4.1. Estadísticas Descriptivas
    for m in models:
        data = df[df["model"] == m]["latency_ms"].values
        mean_val = float(np.mean(data))
        std_val = float(np.std(data, ddof=1))
        median_val = float(np.median(data))
        iqr_val = float(stats.iqr(data))
        ci_low, ci_high = stats.t.interval(1 - alpha, len(data)-1, loc=mean_val, scale=stats.sem(data))
        p95 = float(np.percentile(data, 95))
        p99 = float(np.percentile(data, 99))
        fps = float(1000.0 / mean_val)
        
        results["descriptive"][m] = {
            "model": m,
            "N": len(data),
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "median": round(median_val, 2),
            "iqr": round(iqr_val, 2),
            "min": round(float(np.min(data)), 2),
            "max": round(float(np.max(data)), 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "ci_95_low": round(ci_low, 2),
            "ci_95_high": round(ci_high, 2),
            "fps": round(fps, 1)
        }
        
    # 4.2. Pruebas de Normalidad
    all_normal = True
    for m in models:
        data = df[df["model"] == m]["latency_ms"].values
        shapiro_stat, shapiro_p = stats.shapiro(data)
        ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
        is_normal = bool(shapiro_p >= alpha)
        if not is_normal:
            all_normal = False
        results["normality"][m] = {
            "shapiro_stat": float(shapiro_stat),
            "shapiro_p": float(shapiro_p),
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
            "is_normal": is_normal
        }
        
    # 4.3. Homocedasticidad (Levene)
    levene_stat, levene_p = stats.levene(*groups)
    results["homoscedasticity"] = {
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "equal_variance": bool(levene_p >= alpha)
    }
    
    # 4.4. One-Way ANOVA
    f_stat, anova_p = stats.f_oneway(*groups)
    results["anova"] = {
        "F_statistic": float(f_stat),
        "p_value": float(anova_p),
        "reject_H0": bool(anova_p < alpha)
    }
    
    # 4.5. Tukey HSD Post-hoc
    scipy_tukey = stats.tukey_hsd(*groups)
    ci_tukey = scipy_tukey.confidence_interval(confidence_level=1.0 - alpha)
    
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            diff = float(scipy_tukey.statistic[i, j])
            p_val = float(scipy_tukey.pvalue[i, j])
            reject = bool(p_val < alpha)
            c_low = float(ci_tukey.low[i, j])
            c_high = float(ci_tukey.high[i, j])
            results["tukey_hsd"].append({
                "group1": m1,
                "group2": m2,
                "mean_diff": round(diff, 3),
                "p_adj": float(p_val),
                "reject": reject,
                "ci_lower": round(c_low, 3),
                "ci_upper": round(c_high, 3)
            })
            
    # 4.6. Kruskal-Wallis & Mann-Whitney U con corrección de Bonferroni
    kw_stat, kw_p = stats.kruskal(*groups)
    results["kruskal_wallis"] = {
        "H_statistic": float(kw_stat),
        "p_value": float(kw_p),
        "reject_H0": bool(kw_p < alpha)
    }
    
    num_comparisons = len(models) * (len(models) - 1) // 2
    alpha_bonferroni = alpha / num_comparisons
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            u_stat, u_p = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            results["mann_whitney_bonferroni"].append({
                "group1": m1,
                "group2": m2,
                "U_statistic": float(u_stat),
                "raw_p_value": float(u_p),
                "p_adj_bonferroni": float(min(1.0, u_p * num_comparisons)),
                "reject": bool(u_p < alpha_bonferroni)
            })
            
    # 4.7. Análisis de Aceleración ONNX Runtime vs PyTorch CPU (Speedup)
    common_pairs = [
        ("YOLOv5-nu", "YOLOv5-nu [PyTorch]", "YOLOv5-nu [ONNX]"),
        ("YOLOv8-n", "YOLOv8-n [PyTorch]", "YOLOv8-n [ONNX]"),
        ("YOLO11-n", "YOLO11-n [PyTorch]", "YOLO11-n [ONNX]"),
        ("YOLO26-n-seg", "YOLO26-n-seg [PyTorch]", "YOLO26-n-seg (Nidos) [ONNX]")
    ]
    for label, pytorch_key, onnx_key in common_pairs:
        if pytorch_key in results["descriptive"] and onnx_key in results["descriptive"]:
            t_pt = results["descriptive"][pytorch_key]["mean"]
            t_onnx = results["descriptive"][onnx_key]["mean"]
            speedup = t_pt / t_onnx
            reduction_pct = ((t_pt - t_onnx) / t_pt) * 100.0
            results["speedup_onnx_vs_pytorch"][label] = {
                "pytorch_mean_ms": t_pt,
                "onnx_mean_ms": t_onnx,
                "speedup_factor": round(speedup, 2),
                "latency_reduction_pct": round(reduction_pct, 2)
            }
            
    # 4.8. Verificación de la Hipótesis Principal
    target_model = "YOLO26-n (Turtle v2) [ONNX]"
    target_mean = results["descriptive"][target_model]["mean"]
    onnx_competitors = ["YOLOv5-nu [ONNX]", "YOLOv8-n [ONNX]", "YOLO11-n [ONNX]"]
    
    comparisons_summary = []
    target_faster_all = True
    for comp in onnx_competitors:
        comp_mean = results["descriptive"][comp]["mean"]
        tukey_entry = next((item for item in results["tukey_hsd"] 
                            if (item["group1"] == target_model and item["group2"] == comp) or 
                               (item["group2"] == target_model and item["group1"] == comp)), None)
        is_sig = tukey_entry["reject"] if tukey_entry else False
        p_val = tukey_entry["p_adj"] if tukey_entry else 1.0
        diff_pct = ((comp_mean - target_mean) / comp_mean) * 100.0
        is_faster = (target_mean < comp_mean)
        if not (is_faster and is_sig):
            target_faster_all = False
            
        comparisons_summary.append({
            "target": target_model,
            "competitor": comp,
            "target_mean_ms": target_mean,
            "competitor_mean_ms": comp_mean,
            "speedup_pct": round(diff_pct, 2),
            "p_adj": p_val,
            "statistically_significant": is_sig,
            "validated": is_faster and is_sig
        })
        
    results["hypothesis_decision"] = {
        "target_model": target_model,
        "global_h0_rejected": bool(anova_p < alpha or kw_p < alpha),
        "target_is_significantly_faster_than_all_competitors": target_faster_all,
        "comparisons": comparisons_summary
    }
    
    return results, scipy_tukey


# ===============================================================================================
# 5. GENERACIÓN DE FIGURAS CIENTÍFICAS DE ALTA RESOLUCIÓN (300 DPI)
# ===============================================================================================
def generate_scientific_visualizations(df: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
    """Crea visualizaciones académicas preparadas para publicación en la JIC 2026."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    models = results["models"]
    
    # -------------------------------------------------------------------------------------------
    # FIGURA 1: Boxplot Comparativo de Todos los Modelos (11 Modelos)
    # -------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    data_by_model = [df[df["model"] == m]["latency_ms"].values for m in models]
    
    # Colores diferenciados: Verde para YOLO26 del proyecto, Azul para Competidores ONNX, Magenta para PyTorch
    box_colors = []
    for m in models:
        if "turtle v2" in m.lower():
            box_colors.append("#2ca02c")
        elif "yolo26" in m.lower():
            box_colors.append("#3cb44b" if "onnx" in m.lower() else "#4363d8")
        elif "onnx" in m.lower():
            box_colors.append("#1f77b4")
        else:
            box_colors.append("#e6194b")
            
    bplot = ax.boxplot(
        data_by_model,
        patch_artist=True,
        labels=[m.replace(" [ONNX]", "\n(ONNX)").replace(" [PyTorch]", "\n(PyTorch)") for m in models],
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 7},
        medianprops={"color": "black", "linewidth": 2}
    )
    
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        
    # Jitter de puntos individuales
    for idx, d in enumerate(data_by_model):
        x = np.random.normal(idx + 1, 0.08, size=len(d))
        ax.scatter(x, d, alpha=0.25, color="black", s=10)
        
    # Línea umbral de tiempo real (20 FPS = 50 ms)
    ax.axhline(50.0, color="crimson", linestyle="--", linewidth=1.5, label="Umbral Tiempo Real (20 FPS / 50 ms)")
    
    ax.set_title("Comparación Exhaustiva de Latencia de Inferencia en CPU (ms)\nModelos de Detección y Segmentación para Monitoreo de Tortugas Marinas con UAV", fontsize=13, weight="bold", pad=12)
    ax.set_ylabel("Latencia de Inferencia en CPU (ms)", fontsize=11, weight="bold")
    ax.set_xlabel("Modelo y Motor de Ejecución", fontsize=11, weight="bold")
    ax.tick_params(axis='x', rotation=35, labelsize=9)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    
    fig1_png = output_dir / "figure1_all_models_cpu_latency.png"
    plt.savefig(fig1_png, dpi=300)
    plt.savefig(output_dir / "figure1_all_models_cpu_latency.pdf")
    plt.close()
    print(f"  [+] Figura 1 guardada: {fig1_png}")
    
    # -------------------------------------------------------------------------------------------
    # FIGURA 2: Comparativa Directa PyTorch CPU vs ONNX Runtime CPU (Aceleración Speedup)
    # -------------------------------------------------------------------------------------------
    speedups = results["speedup_onnx_vs_pytorch"]
    if speedups:
        fig, (ax_lat, ax_fps) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        archs = list(speedups.keys())
        x = np.arange(len(archs))
        width = 0.35
        
        pt_lat = [speedups[k]["pytorch_mean_ms"] for k in archs]
        onnx_lat = [speedups[k]["onnx_mean_ms"] for k in archs]
        
        b1 = ax_lat.bar(x - width/2, pt_lat, width, label='PyTorch CPU Nativo', color='#e6194b', alpha=0.85)
        b2 = ax_lat.bar(x + width/2, onnx_lat, width, label='ONNX Runtime CPU', color='#2ca02c', alpha=0.85)
        
        ax_lat.set_ylabel('Latencia Media (ms)', fontsize=11, weight='bold')
        ax_lat.set_title('Impacto de la Optimización de Runtime\nPyTorch CPU vs ONNX Runtime CPU', fontsize=12, weight='bold')
        ax_lat.set_xticks(x)
        ax_lat.set_xticklabels(archs, fontsize=10, weight='bold')
        ax_lat.legend()
        
        for rect in b1 + b2:
            height = rect.get_height()
            ax_lat.annotate(f'{height:.1f}ms',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, weight='bold')
                            
        # Speedup Factors
        speedup_vals = [speedups[k]["speedup_factor"] for k in archs]
        colors_sp = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
        bars_sp = ax_fps.bar(archs, speedup_vals, color=colors_sp, alpha=0.85, width=0.5)
        ax_fps.axhline(1.0, color='gray', linestyle='--', linewidth=1.2)
        ax_fps.set_ylabel('Factor de Aceleración (Speedup $X$)', fontsize=11, weight='bold')
        ax_fps.set_title('Ganancia de Velocidad al Desplegar en ONNX Runtime\n(Speedup = Latencia PyTorch / Latencia ONNX)', fontsize=12, weight='bold')
        
        for bar in bars_sp:
            yval = bar.get_height()
            ax_fps.annotate(f'{yval:.2f}x\n(+{((yval-1)*100):.0f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, yval),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, weight='bold')
                            
        plt.tight_layout()
        fig2_png = output_dir / "figure2_onnx_vs_pytorch_speedup.png"
        plt.savefig(fig2_png, dpi=300)
        plt.savefig(output_dir / "figure2_onnx_vs_pytorch_speedup.pdf")
        plt.close()
        print(f"  [+] Figura 2 guardada: {fig2_png}")
        
    # -------------------------------------------------------------------------------------------
    # FIGURA 3: Forest Plot - Intervalos de Confianza al 95%
    # -------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    means = [results["descriptive"][m]["mean"] for m in models]
    ci_lows = [results["descriptive"][m]["mean"] - results["descriptive"][m]["ci_95_low"] for m in models]
    ci_highs = [results["descriptive"][m]["ci_95_high"] - results["descriptive"][m]["mean"] for m in models]
    y_pos = np.arange(len(models))
    
    ax.errorbar(means, y_pos, xerr=[ci_lows, ci_highs], fmt='o', color='navy', ecolor='teal', elinewidth=2.2, capsize=5, markersize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=9, weight='bold')
    ax.axvline(50.0, color="crimson", linestyle="--", label="Límite 20 FPS (50 ms)")
    ax.set_xlabel("Latencia Media en CPU (ms) [IC 95%]", fontsize=11, weight="bold")
    ax.set_title("Intervalos de Confianza al 95% de la Latencia Media en CPU\n(Estimación Incondicional por Modelo)", fontsize=13, weight="bold", pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig3_png = output_dir / "figure3_tukey_hsd_intervals.png"
    plt.savefig(fig3_png, dpi=300)
    plt.savefig(output_dir / "figure3_tukey_hsd_intervals.pdf")
    plt.close()
    print(f"  [+] Figura 3 guardada: {fig3_png}")


# ===============================================================================================
# 6. EXPORTACIÓN DE TABLAS CIENTÍFICAS
# ===============================================================================================
def export_scientific_tables(df: pd.DataFrame, results: Dict[str, Any], output_dir: Path):
    """Guarda tablas descriptivas, de contrastes de hipótesis y LaTeX."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Datos Crudos
    raw_path = output_dir / "latency_raw_measurements.csv"
    df.to_csv(raw_path, index=False)
    
    # 2. Estadísticos Descriptivos
    desc_df = pd.DataFrame(list(results["descriptive"].values()))
    desc_path = output_dir / "table_descriptive_statistics.csv"
    desc_df.to_csv(desc_path, index=False)
    
    # 3. Tukey Post-hoc
    tukey_df = pd.DataFrame(results["tukey_hsd"])
    tukey_path = output_dir / "table_tukey_hsd_posthoc.csv"
    tukey_df.to_csv(tukey_path, index=False)
    
    # 4. JSON Completo
    json_path = output_dir / "hypothesis_test_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    # 5. Tabla LaTeX Formateada para Artículo de Investigación
    latex_code = desc_df.to_latex(
        index=False,
        columns=["model", "N", "mean", "std", "median", "iqr", "ci_95_low", "ci_95_high", "fps"],
        header=["Modelo", "$N$", "Media (ms)", "DE (ms)", "Mediana (ms)", "IQR (ms)", "IC 95% Inf", "IC 95% Sup", "FPS"],
        caption="Evaluación experimental exhaustiva de latencia de inferencia en CPU para modelos YOLO en monitoreo UAV de tortugas marinas.",
        label="tab:cpu_latency_full_benchmark"
    )
    with open(output_dir / "table_latex_paper.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)
        
    print(f"\n📁 Reportes generados en {output_dir}:")
    print(f"   • {raw_path.name}")
    print(f"   • {desc_path.name}")
    print(f"   • {tukey_path.name}")
    print(f"   • {json_path.name}")
    print(f"   • table_latex_paper.tex")


# ===============================================================================================
# 7. EJECUCIÓN PRINCIPAL
# ===============================================================================================
def main():
    t_start = time.time()
    hw_info = log_hardware_environment()
    
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_records = []
    
    print("="*85)
    print(f"🚀 INICIANDO BATERÍA DE EVALUACIÓN: {len(MODEL_EXPERIMENTS)} MODELOS / CONFIGURACIONES")
    print(f"   • Muestras por modelo: {CONFIG['num_samples']} | Calentamiento: {CONFIG['warmup_runs']}")
    print("="*85)
    
    for idx, exp in enumerate(MODEL_EXPERIMENTS):
        m_id = exp["id"]
        engine = exp["engine"]
        path = exp["path"]
        
        print(f"\n[{idx+1}/{len(MODEL_EXPERIMENTS)}] Evaluando: {m_id}")
        print(f"     Motor: {engine.upper()} | Tarea: {exp['task']} | Archivo: {path}")
        
        if not os.path.isfile(path):
            print(f"     ❌ Archivo no encontrado: {path}. Saltando...")
            continue
            
        if engine == "onnx":
            latencies = run_benchmark_onnx(
                model_name=m_id,
                onnx_path=path,
                num_samples=CONFIG["num_samples"],
                warmup_runs=CONFIG["warmup_runs"],
                img_size=CONFIG["img_size"]
            )
        else:
            latencies = run_benchmark_pytorch(
                model_name=m_id,
                weight_path=path,
                num_samples=CONFIG["num_samples"],
                warmup_runs=CONFIG["warmup_runs"],
                img_size=CONFIG["img_size"]
            )
            
        mean_l = np.mean(latencies)
        print(f"     ✅ Media: {mean_l:.2f} ms | FPS: {1000.0/mean_l:.1f} FPS | Min: {np.min(latencies):.2f} ms | Max: {np.max(latencies):.2f} ms")
        
        for s_idx, lat in enumerate(latencies):
            all_records.append({
                "model": m_id,
                "engine": engine,
                "task": exp["task"],
                "is_project": exp["is_project"],
                "iteration": s_idx + 1,
                "latency_ms": lat
            })
            
    df = pd.DataFrame(all_records)
    
    # Análisis Estadístico
    results, scipy_tukey = analyze_statistics(df, alpha=CONFIG["alpha"])
    results["hardware"] = hw_info
    results["config"] = CONFIG
    
    # Visualizaciones y Tablas
    print("\n📊 Generando figuras de alta resolución y tablas estadísticas...")
    generate_scientific_visualizations(df, results, output_dir)
    export_scientific_tables(df, results, output_dir)
    
    # Espejar a Downloads si existe la ruta
    dl_dir = Path(CONFIG["downloads_mirror"])
    try:
        dl_dir.mkdir(parents=True, exist_ok=True)
        for item in output_dir.glob("*.*"):
            shutil.copy2(item, dl_dir / item.name)
        print(f"📂 Archivos espejados exitosamente a: {dl_dir}")
    except Exception as e:
        print(f"⚠️ Nota: No se pudo espejar a descargas ({e})")
        
    elapsed = time.time() - t_start
    print("\n" + "="*85)
    print(f"🎉 EXPERIMENTO COMPLETO FINALIZADO EN {elapsed:.1f} SEGUNDOS")
    print(f"   Total de mediciones cronometradas: {len(df):,} inferencias")
    print("="*85)


if __name__ == "__main__":
    main()
