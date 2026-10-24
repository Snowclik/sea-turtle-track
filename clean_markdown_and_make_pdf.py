#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para depurar el formato Markdown y generar un PDF con estilo editorial académico sobrio
(LaTeX Article / Booktabs), garantizando que ninguna figura genere hojas en blanco.
"""

import re
import base64
import shutil
from pathlib import Path
from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright

def clean_markdown_text(text: str) -> str:
    # 1. Corregir etiquetas malinterpretadas como HTML (< 10^-15)
    text = text.replace('**$< 10^{-15}$**', '**p < 10⁻¹⁵**')
    text = text.replace('**$< 10^{-14}$**', '**p < 10⁻¹⁴**')
    text = text.replace('$< 10^{-15}$', 'p < 10⁻¹⁵')
    text = text.replace('$< 10^{-14}$', 'p < 10⁻¹⁴')
    text = text.replace('$p < 10^{-100}$', 'p < 10⁻¹⁰⁰')
    text = text.replace('$p < 10^{-200}$', 'p < 10⁻²⁰⁰')
    text = text.replace('$p < 10^{-12}$', 'p < 10⁻¹²')
    text = text.replace('$p < 10^{-18}$', 'p < 10⁻¹⁸')
    text = text.replace('$p < 10^{-9}$', 'p < 10⁻⁹')
    text = text.replace('$p < 0.001$', 'p < 0.001')
    text = text.replace('$p > 0.05$', 'p > 0.05')
    text = text.replace('< 10^{-15}', 'p < 10⁻¹⁵')
    text = text.replace('< 10^{-14}', 'p < 10⁻¹⁴')
    text = text.replace('< 10^{-100}', 'p < 10⁻¹⁰⁰')
    text = text.replace('< 10^{-200}', 'p < 10⁻²⁰⁰')
    text = text.replace('{,}', ',')

    # 2. Reemplazos de comandos LaTeX crudos
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*\\text\{\s*ms\}\$', r'\1 ms', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*\\text\{\s*FPS\}\$', r'\1 FPS', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*\\text\{\s*W\}\$', r'\1 W', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*\\text\{\s*m/s\}\$', r'\1 m/s', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*\\text\{\s*km/h\}\$', r'\1 km/h', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*\\text\{\s*segundos\}\$', r'\1 s', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\s*\\text\{\s*metros\}\$', r'\1 m', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)\$', r'\1', text)

    text = text.replace('$\\mu$', 'μ')
    text = text.replace('$\\sigma$', 'σ')
    text = text.replace('$\\Delta \\mu$', 'Δμ')
    text = text.replace('$\\Delta =', 'Δ =')
    text = text.replace('$\\alpha$', 'α')
    text = text.replace('$\\alpha_{\\text{adj}}', 'α_adj')
    text = text.replace('\\mu', 'μ')
    text = text.replace('\\sigma', 'σ')
    text = text.replace('\\Delta', 'Δ')
    text = text.replace('\\le', '≤')
    text = text.replace('\\ge', '≥')
    text = text.replace('$\\ge', '≥')
    text = text.replace('$\\le', '≤')
    text = text.replace('\\times', '×')
    text = text.replace('p_{adj}', 'p_adj')
    text = text.replace('{adj}', '_adj')

    # Limpiar fórmulas de bloque
    text = text.replace('$$\\text{FPS} = \\frac{1000}{\\text{Latencia Media en ms}}$$', '**FPS = 1000 / Latencia Media (ms)**')
    text = text.replace('$$\\text{Speedup} = \\frac{\\text{Latencia}_{\\text{PyTorch}}}{\\text{Latencia}_{\\text{ONNX}}}$$', '**Factor de Aceleración (Speedup) = Latencia PyTorch / Latencia ONNX**')

    # Limpiar comandos \text restantes y signos de dólar aislados
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
    text = text.replace('$', '')
    
    return text


def build_academic_html(body_html: str) -> str:
    """Genera HTML con diseño editorial sobrio estilo LaTeX Article / IEEE."""
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe Científico - Validación de Hipótesis CPU</title>
<style>
    @page {{
        size: A4;
        margin: 20mm 18mm 20mm 18mm;
        @bottom-center {{
            content: "- " counter(page) " -";
            font-family: 'Times New Roman', Times, 'Nimbus Roman No9 L', serif;
            font-size: 8.5pt;
            color: #444444;
        }}
    }}
    body {{
        font-family: 'Times New Roman', Times, 'Nimbus Roman No9 L', serif;
        color: #111111;
        line-height: 1.45;
        font-size: 10pt;
        background-color: #ffffff;
    }}
    
    /* Encabezados estilo LaTeX Article */
    h1 {{
        font-size: 14pt;
        font-weight: bold;
        text-align: center;
        color: #000000;
        text-transform: uppercase;
        letter-spacing: 0.5pt;
        margin-top: 12pt;
        margin-bottom: 10pt;
        page-break-after: avoid;
        break-after: avoid;
    }}
    h2 {{
        font-size: 11.5pt;
        font-weight: bold;
        color: #000000;
        border-bottom: 0.75pt solid #000000;
        padding-bottom: 2pt;
        margin-top: 14pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
        break-after: avoid;
    }}
    h3 {{
        font-size: 10.5pt;
        font-weight: bold;
        color: #111111;
        margin-top: 10pt;
        margin-bottom: 4pt;
        page-break-after: avoid;
        break-after: avoid;
    }}
    h4 {{
        font-size: 10pt;
        font-style: italic;
        font-weight: bold;
        color: #222222;
        margin-top: 8pt;
        margin-bottom: 3pt;
        page-break-after: avoid;
        break-after: avoid;
    }}
    
    p {{
        margin-top: 3pt;
        margin-bottom: 5pt;
        text-align: justify;
        text-indent: 14pt;
    }}
    
    h1 + p, h2 + p, h3 + p, h4 + p, hr + p, blockquote + p, table + p {{
        text-indent: 0;
    }}
    
    ul, ol {{
        margin-top: 3pt;
        margin-bottom: 5pt;
        padding-left: 20pt;
        text-align: justify;
    }}
    li {{
        margin-bottom: 2pt;
    }}
    
    /* Tablas estilo LaTeX Booktabs */
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 8pt;
        margin-bottom: 10pt;
        font-size: 7.8pt;
        line-height: 1.25;
        border-top: 1.5pt solid #000000;     /* toprule */
        border-bottom: 1.5pt solid #000000;  /* bottomrule */
        page-break-inside: avoid;
        break-inside: avoid;
    }}
    th {{
        font-weight: bold;
        color: #000000;
        border-bottom: 1pt solid #000000;   /* midrule */
        padding: 3.5pt 4pt;
        text-align: left;
        background-color: transparent;
    }}
    td {{
        padding: 3pt 4pt;
        text-align: left;
        border: none;
        border-bottom: 0.5pt solid #e2e2e2;
        background-color: transparent;
    }}
    tr:last-child td {{
        border-bottom: none;
    }}
    
    /* Citas y Párrafos Destacados */
    blockquote {{
        border-left: 2pt solid #444444;
        padding: 4pt 10pt;
        margin: 6pt 12pt;
        background-color: #fafafa;
        font-size: 9.5pt;
        text-align: justify;
        page-break-inside: avoid;
        break-inside: avoid;
    }}
    blockquote p {{
        text-indent: 0;
        margin: 2pt 0;
    }}
    
    /* Figuras e Ilustraciones académicas ajustadas a la página */
    img {{
        max-width: 92%;
        max-height: 145mm;
        object-fit: contain;
        display: block;
        margin: 6pt auto 3pt auto;
        border: 0.5pt solid #cccccc;
        page-break-inside: avoid;
        break-inside: avoid;
    }}
    em {{
        font-size: 8pt;
        font-style: italic;
        color: #333333;
        display: block;
        text-align: center;
        margin-top: 2pt;
        margin-bottom: 8pt;
        page-break-before: avoid;
        break-before: avoid;
    }}
    
    hr {{
        border: none;
        border-top: 0.5pt solid #cccccc;
        margin: 10pt 0;
    }}
    code {{
        font-family: 'Courier New', Courier, monospace;
        font-size: 8pt;
        background-color: #f4f4f4;
        padding: 1pt 2pt;
    }}
</style>
</head>
<body>
{body_html}
</body>
</html>"""


def run_pipeline():
    md_file = Path("informe_validacion_hipotesis_cpu.md")
    with open(md_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    clean_text = clean_markdown_text(raw_text)

    # Guardar markdown limpio en el repo y en la carpeta V2
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(clean_text)

    v2_dir = Path("C:/Users/tavog/Downloads/hypothesis_latency_cpu/V2")
    v2_dir.mkdir(parents=True, exist_ok=True)
    with open(v2_dir / "informe_validacion_hipotesis_cpu.md", "w", encoding="utf-8") as f:
        f.write(clean_text)

    # Reemplazar imágenes con Base64 para autosuficiencia total
    def img_replacer(match):
        alt = match.group(1)
        img_rel = match.group(2).lstrip("./")
        img_path = Path(img_rel)
        if not img_path.is_file():
            img_path = Path("reports/hypothesis_latency_cpu") / img_path.name
        if img_path.is_file():
            with open(img_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
            ext = img_path.suffix.lstrip(".").lower()
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            return f"![{alt}](data:{mime};base64,{b64})"
        return match.group(0)

    text_b64 = re.sub(r"!\[(.*?)\]\((.*?)\)", img_replacer, clean_text)

    md = MarkdownIt("commonmark").enable("table")
    body_html = md.render(text_b64)
    academic_html = build_academic_html(body_html)

    tmp_html = Path("academic_report_temp.html")
    with open(tmp_html, "w", encoding="utf-8") as f:
        f.write(academic_html)

    out_pdf_repo = Path("informe_validacion_hipotesis_cpu.pdf")
    out_pdf_v2 = v2_dir / "informe_validacion_hipotesis_cpu.pdf"

    print("Compilando PDF en estilo formal de artículo científico (LaTeX Style)...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(tmp_html.resolve().as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(out_pdf_repo),
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "18mm", "right": "18mm"}
        )
        browser.close()

    shutil.copy2(out_pdf_repo, out_pdf_v2)
    if tmp_html.exists():
        tmp_html.unlink()

    print(f"[EXITO] PDF generado en repo: {out_pdf_repo} ({out_pdf_repo.stat().st_size:,} bytes)")
    print(f"[EXITO] PDF copiado a V2 : {out_pdf_v2} ({out_pdf_v2.stat().st_size:,} bytes)")

if __name__ == "__main__":
    run_pipeline()
