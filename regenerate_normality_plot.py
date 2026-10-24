#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

def generate_compact_normality_figure():
    df = pd.read_csv("reports/hypothesis_latency_cpu/latency_raw_measurements.csv")
    
    # 6 Modelos clave para diagnóstico de normalidad en ONNX (4 del proyecto + 2 competidores directos)
    models_to_plot = [
        ("YOLO26-n (Tortugas v3) [ONNX]", "YOLO26-n v3 (Detección)"),
        ("YOLO26-n (Turtle v2) [ONNX]", "YOLO26-n v2 (Producción)"),
        ("YOLO11-n [ONNX]", "YOLO11-n (Competidor Directo)"),
        ("YOLOv5-nu [ONNX]", "YOLOv5-nu (Baseline)"),
        ("YOLOv8-n [ONNX]", "YOLOv8-n (Baseline)"),
        ("YOLO26-n-seg (Nidos) [ONNX]", "YOLO26-n-seg (Segmentación)")
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.2), dpi=300)
    plt.subplots_adjust(hspace=0.46, wspace=0.30, left=0.06, right=0.96, top=0.89, bottom=0.08)
    
    fig.suptitle("Diagnóstico de Distribución y Normalidad de la Latencia en CPU (Histogramas KDE y Gráficos Q-Q)",
                 fontsize=12, weight="bold", y=0.96)

    for idx, (m_id, label) in enumerate(models_to_plot):
        row = idx // 2
        col_base = (idx % 2) * 2
        
        data = df[df["model"] == m_id]["latency_ms"].values
        sw_stat, sw_p = stats.shapiro(data)
        
        # 1. Histograma con curva Normal teórica
        ax_hist = axes[row, col_base]
        n_bins = 15
        counts, bins, _ = ax_hist.hist(data, bins=n_bins, density=True, color="#2b6cb0", alpha=0.6, edgecolor="#1a365d")
        
        # Ajuste curva normal
        mu, std = np.mean(data), np.std(data, ddof=1)
        x_norm = np.linspace(min(data), max(data), 100)
        p_norm = stats.norm.pdf(x_norm, mu, std)
        ax_hist.plot(x_norm, p_norm, color="#e53e3e", linewidth=1.5, linestyle="--")
        
        ax_hist.set_title(f"{label}\nShapiro p={sw_p:.2e}", fontsize=8.5, weight="bold")
        ax_hist.set_xlabel("Latencia (ms)", fontsize=7.5)
        ax_hist.set_ylabel("Densidad", fontsize=7.5)
        ax_hist.tick_params(labelsize=7)
        ax_hist.grid(True, linestyle=":", alpha=0.6)
        
        # 2. Q-Q Plot Normal
        ax_qq = axes[row, col_base + 1]
        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
        ax_qq.plot(osm, osr, "o", color="#1a365d", markersize=3.5, alpha=0.7)
        ax_qq.plot(osm, slope * np.array(osm) + intercept, color="#e53e3e", linewidth=1.5)
        
        ax_qq.set_title(f"Q-Q Plot: {label}\nR² = {r**2:.3f}", fontsize=8.5, weight="bold")
        ax_qq.set_xlabel("Cuantiles Teóricos", fontsize=7.5)
        ax_qq.set_ylabel("Valores Muestra (ms)", fontsize=7.5)
        ax_qq.tick_params(labelsize=7)
        ax_qq.grid(True, linestyle=":", alpha=0.6)
        
    out_paths = [
        Path("reports/hypothesis_latency_cpu/figure2_normality_diagnostics.png"),
        Path("figure2_normality_diagnostics.png"),
        Path("C:/Users/tavog/Downloads/hypothesis_latency_cpu/V2/figure2_normality_diagnostics.png")
    ]
    
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
        print(f"[OK] Saved compact normality figure: {p}")
        
    plt.close()

if __name__ == "__main__":
    generate_compact_normality_figure()
