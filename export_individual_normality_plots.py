#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para generar y exportar de forma individual las 12 gráficas de diagnóstico 
distribucional y normalidad (Histograma con ajuste normal + Q-Q Plot para cada uno 
de los 6 modelos evaluados en ONNX Runtime CPU).

Destino:
- C:/Users/tavog/Downloads/hypothesis_latency_cpu/V2/grafica/
- reports/hypothesis_latency_cpu/grafica/
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# Configuración estética formal y académica
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def export_individual_plots():
    csv_path = Path("reports/hypothesis_latency_cpu/latency_raw_measurements.csv")
    if not csv_path.is_file():
        csv_path = Path("latency_raw_measurements.csv")
    df = pd.read_csv(csv_path)

    # Definición de los 6 modelos clave y su denominación académica
    models_config = [
        {
            "id": "YOLO26-n (Tortugas v3) [ONNX]",
            "slug": "01_yolo26n_v3_final",
            "name": "YOLO26-n v3 (Versión Final Oficial)",
            "task": "Detección de Rastros (Dataset 2,515 imgs)"
        },
        {
            "id": "YOLO26-n (Turtle v2) [ONNX]",
            "slug": "02_yolo26n_v2_previa",
            "name": "YOLO26-n v2 (Versión Previa Experimental)",
            "task": "Detección de Rastros (Base + Naturaleza)"
        },
        {
            "id": "YOLO11-n [ONNX]",
            "slug": "03_yolo11n_competidor",
            "name": "YOLO11-n (Competidor Directo)",
            "task": "Detección Nano de Última Generación"
        },
        {
            "id": "YOLOv5-nu [ONNX]",
            "slug": "04_yolov5nu_baseline",
            "name": "YOLOv5-nu (Línea Base)",
            "task": "Detección Nano Clásica"
        },
        {
            "id": "YOLOv8-n [ONNX]",
            "slug": "05_yolov8n_baseline",
            "name": "YOLOv8-n (Línea Base)",
            "task": "Detección Nano Predecesora"
        },
        {
            "id": "YOLO26-n-seg (Nidos) [ONNX]",
            "slug": "06_yolo26n_seg_nidos",
            "name": "YOLO26-n-seg (Segmentación de Nidos)",
            "task": "Segmentación de Instancias"
        }
    ]

    out_dirs = [
        Path("C:/Users/tavog/Downloads/hypothesis_latency_cpu/V2/grafica"),
        Path("reports/hypothesis_latency_cpu/grafica")
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    generated_files = []
    summary_data = []

    plot_counter = 1

    for m in models_config:
        model_id = m["id"]
        model_name = m["name"]
        model_slug = m["slug"]
        model_task = m["task"]

        data = df[df["model"] == model_id]["latency_ms"].values
        n_samples = len(data)
        mu = np.mean(data)
        std = np.std(data, ddof=1)
        median = np.median(data)
        iqr = stats.iqr(data)

        # Prueba Shapiro-Wilk
        sw_stat, sw_p = stats.shapiro(data)
        dist_desc = "Normal" if sw_p >= 0.05 else "No Normal (p < 0.05)"

        # -------------------------------------------------------------
        # 1. GRÁFICA A: HISTOGRAMA CON AJUSTE DE DISTRIBUCIÓN NORMAL
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=300)

        # Histograma empírico
        n_bins = 15
        counts, bins, patches = ax.hist(
            data, bins=n_bins, density=True, 
            color="#2b6cb0", alpha=0.65, edgecolor="#1a365d", linewidth=1.0,
            label=f"Datos Observados (N={n_samples})"
        )

        # Curva de ajuste normal teórico N(mu, std)
        x_norm = np.linspace(min(data) - 0.05 * (max(data) - min(data)), 
                             max(data) + 0.05 * (max(data) - min(data)), 200)
        p_norm = stats.norm.pdf(x_norm, mu, std)
        ax.plot(x_norm, p_norm, color="#e53e3e", linewidth=2.0, linestyle="--", 
                label=f"Ajuste Normal Teórico: N({mu:.2f}, {std:.2f})")

        # Línea de media y mediana
        ax.axvline(mu, color="#2c5282", linestyle="-", linewidth=1.4, alpha=0.9, label=f"Media (μ = {mu:.2f} ms)")
        ax.axvline(median, color="#234e52", linestyle=":", linewidth=1.4, alpha=0.9, label=f"Mediana = {median:.2f} ms")

        # Título y textos
        ax.set_title(f"{model_name}\nHistograma de Latencia y Ajuste Normal (Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.2e})", 
                     fontsize=10, weight="bold", pad=10)
        ax.set_xlabel("Latencia de Inferencia en CPU (ms)", fontsize=9, labelpad=6)
        ax.set_ylabel("Densidad de Probabilidad", fontsize=9, labelpad=6)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.55, color="#a0aec0")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)

        # Cuadro informativo lateral
        info_box = (f"N = {n_samples}\n"
                    f"μ = {mu:.2f} ms\n"
                    f"σ = {std:.2f} ms\n"
                    f"Mediana = {median:.2f} ms\n"
                    f"IQR = {iqr:.2f} ms\n"
                    f"Mín = {np.min(data):.2f} ms\n"
                    f"Máx = {np.max(data):.2f} ms\n"
                    f"Shapiro W = {sw_stat:.4f}\n"
                    f"p-valor = {sw_p:.2e}\n"
                    f"Diagnóstico: {dist_desc}")
        ax.text(0.03, 0.95, info_box, transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7fafc', edgecolor='#cbd5e0', alpha=0.9))

        plt.tight_layout()

        hist_filename_png = f"grafica_{plot_counter:02d}_histograma_kde_{model_slug}.png"
        hist_filename_pdf = f"grafica_{plot_counter:02d}_histograma_kde_{model_slug}.pdf"

        for d in out_dirs:
            plt.savefig(d / hist_filename_png, dpi=300)
            plt.savefig(d / hist_filename_pdf)
        plt.close()

        generated_files.append((plot_counter, hist_filename_png, f"Histograma de distribución y prueba de Shapiro-Wilk para {model_name}"))
        plot_counter += 1

        # -------------------------------------------------------------
        # 2. GRÁFICA B: GRÁFICO DE PROBABILIDAD NORMAL (Q-Q PLOT)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6.2, 4.5), dpi=300)

        (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")
        r_squared = r**2

        # Puntos de dispersión empíricos
        ax.plot(osm, osr, "o", color="#1a365d", markersize=4.5, alpha=0.75, 
                markeredgecolor="#0f2942", label=f"Cuantiles Muestrales (N={n_samples})")

        # Línea teórica de referencia normal
        ax.plot(osm, slope * np.array(osm) + intercept, color="#e53e3e", linewidth=2.0, 
                label=f"Recta de Referencia Normal: y = {slope:.2f}x + {intercept:.2f}")

        ax.set_title(f"Gráfico Q-Q Normal: {model_name}\nCoeficiente de Determinación R² = {r_squared:.4f}", 
                     fontsize=10, weight="bold", pad=10)
        ax.set_xlabel("Cuantiles Teóricos (Distribución Normal Estándar)", fontsize=9, labelpad=6)
        ax.set_ylabel("Valores Muestrales de Latencia (ms)", fontsize=9, labelpad=6)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.55, color="#a0aec0")
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

        # Cuadro informativo de ajuste
        qq_info = (f"R² = {r_squared:.4f}\n"
                   f"Pendiente = {slope:.2f}\n"
                   f"Intercepto = {intercept:.2f}\n"
                   f"Correlación r = {r:.4f}\n"
                   f"Alineación: {'Excelente (Normal)' if r_squared > 0.95 else 'Asimétrica / Outliers'}")
        ax.text(0.03, 0.95, qq_info, transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7fafc', edgecolor='#cbd5e0', alpha=0.9))

        plt.tight_layout()

        qq_filename_png = f"grafica_{plot_counter:02d}_qqplot_{model_slug}.png"
        qq_filename_pdf = f"grafica_{plot_counter:02d}_qqplot_{model_slug}.pdf"

        for d in out_dirs:
            plt.savefig(d / qq_filename_png, dpi=300)
            plt.savefig(d / qq_filename_pdf)
        plt.close()

        generated_files.append((plot_counter, qq_filename_png, f"Gráfico Q-Q Normal de probabilidad y R² para {model_name}"))
        plot_counter += 1

        summary_data.append({
            "Modelo": model_name,
            "Media (ms)": f"{mu:.2f}",
            "Desv. Est. (ms)": f"{std:.2f}",
            "Shapiro W": f"{sw_stat:.4f}",
            "p-valor Shapiro": f"{sw_p:.2e}",
            "R² Q-Q": f"{r_squared:.4f}",
            "Conclusión": dist_desc
        })

    # Crear archivo README / Índice detallado dentro de la carpeta grafica
    readme_content = "# ÍNDICE DE LAS 12 GRÁFICAS INDIVIDUALES DE DIAGNÓSTICO DE NORMALIDAD Y DISTRIBUCIÓN\n\n"
    readme_content += "Este directorio contiene la descomposición individual de las 12 gráficas correspondientes al diagnóstico distribucional de latencia en CPU (Histogramas con ajuste normal y Gráficos Q-Q).\n\n"
    readme_content += "Cada gráfica se encuentra disponible en formato PNG de alta resolución (300 DPI) y en formato vectorial PDF listo para publicaciones, reportes y presentaciones académicas.\n\n"
    readme_content += "## Resumen Estadístico de los Modelos Analizados:\n\n"
    readme_content += "| Modelo | Media (ms) | Desv. Est. (ms) | Shapiro-Wilk (W) | p-valor | R² (Q-Q Plot) | Conclusión Distribucional |\n"
    readme_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for s in summary_data:
        readme_content += f"| **{s['Modelo']}** | {s['Media (ms)']} | {s['Desv. Est. (ms)']} | {s['Shapiro W']} | {s['p-valor Shapiro']} | {s['R² Q-Q']} | {s['Conclusión']} |\n"
    
    readme_content += "\n## Lista de Archivos Generados (12 Gráficas Individuales):\n\n"
    for num, fname, desc in generated_files:
        readme_content += f"{num:02d}. **`{fname}`**  \n   *Descripción:* {desc}  \n   *Versión Vectorial:* `{fname.replace('.png', '.pdf')}`\n\n"

    for d in out_dirs:
        with open(d / "README_graficas.md", "w", encoding="utf-8") as f:
            f.write(readme_content)

    print(f"[EXITO TOTAL] Se exportaron exitosamente las 12 gráficas individuales en PNG y PDF a:")
    for d in out_dirs:
        print(f" - {d.resolve()}")

if __name__ == "__main__":
    export_individual_plots()
