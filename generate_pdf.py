#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import base64
import shutil
from pathlib import Path
from markdown_it import MarkdownIt
from playwright.sync_api import sync_playwright

def convert_md_to_pdf():
    md_path = Path("informe_validacion_hipotesis_cpu.md")
    if not md_path.exists():
        print(f"Error: {md_path} not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Convert image references to embedded Base64 data URIs
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

    text_b64 = re.sub(r"!\[(.*?)\]\((.*?)\)", img_replacer, text)

    md = MarkdownIt("commonmark").enable("table")
    body_html = md.render(text_b64)

    full_html = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe Científico - Validación de Hipótesis CPU</title>
<style>
    @page {
        size: A4;
        margin: 20mm 18mm 22mm 18mm;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1a1a1a;
        line-height: 1.6;
        font-size: 10.5pt;
    }
    h1 {
        font-size: 17pt;
        color: #0d233a;
        border-bottom: 2px solid #0d233a;
        padding-bottom: 4px;
        margin-top: 18pt;
        margin-bottom: 10pt;
        page-break-after: avoid;
    }
    h2 {
        font-size: 13.5pt;
        color: #1a365d;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 3px;
        margin-top: 16pt;
        margin-bottom: 8pt;
        page-break-after: avoid;
    }
    h3 {
        font-size: 11.5pt;
        color: #2b6cb0;
        margin-top: 12pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
    }
    h4 {
        font-size: 10.5pt;
        color: #2d3748;
        margin-top: 10pt;
        margin-bottom: 4pt;
        page-break-after: avoid;
    }
    p, ul, ol {
        margin-top: 4pt;
        margin-bottom: 8pt;
        text-align: justify;
    }
    li {
        margin-bottom: 3pt;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10pt;
        margin-bottom: 12pt;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #cbd5e0;
        padding: 5pt 6pt;
        text-align: left;
    }
    th {
        background-color: #edf2f7;
        color: #2d3748;
        font-weight: bold;
    }
    tr:nth-child(even) {
        background-color: #f7fafc;
    }
    blockquote {
        border-left: 4px solid #3182ce;
        background-color: #ebf8ff;
        padding: 8pt 12pt;
        margin: 10pt 0;
        font-style: normal;
        border-radius: 0 4pt 4pt 0;
        page-break-inside: avoid;
    }
    img {
        max-width: 95%;
        height: auto;
        display: block;
        margin: 12pt auto;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        page-break-inside: avoid;
    }
    em {
        font-size: 8.5pt;
        color: #4a5568;
        display: block;
        text-align: center;
        margin-top: -6pt;
        margin-bottom: 10pt;
    }
    hr {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 16pt 0;
    }
    code {
        background-color: #edf2f7;
        padding: 1pt 3pt;
        border-radius: 3pt;
        font-family: Consolas, monospace;
        font-size: 8.5pt;
    }
</style>
</head>
<body>
""" + body_html + """
</body>
</html>"""

    html_tmp = Path("report_temp.html")
    with open(html_tmp, "w", encoding="utf-8") as f:
        f.write(full_html)

    out_pdf_repo = Path("informe_validacion_hipotesis_cpu.pdf")
    out_pdf_v2 = Path("C:/Users/tavog/Downloads/hypothesis_latency_cpu/V2/informe_validacion_hipotesis_cpu.pdf")
    out_pdf_v2.parent.mkdir(parents=True, exist_ok=True)

    print("Rendering PDF with Playwright Chromium...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_tmp.resolve().as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(out_pdf_repo),
            format="A4",
            print_background=True,
            margin={"top": "20mm", "bottom": "20mm", "left": "18mm", "right": "18mm"}
        )
        browser.close()

    shutil.copy2(out_pdf_repo, out_pdf_v2)
    if html_tmp.exists():
        html_tmp.unlink()

    print(f"[OK] PDF generated in repo: {out_pdf_repo.resolve()} ({out_pdf_repo.stat().st_size:,} bytes)")
    print(f"[OK] PDF copied to V2: {out_pdf_v2.resolve()} ({out_pdf_v2.stat().st_size:,} bytes)")

if __name__ == "__main__":
    convert_md_to_pdf()
