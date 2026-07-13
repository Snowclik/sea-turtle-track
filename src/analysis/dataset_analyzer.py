import os
import glob
import json
import csv
import hashlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS
from jinja2 import Template
import openpyxl
from rich.console import Console
from rich.progress import track
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

console = Console()

# Configuración de directorios
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_IMG_DIR = os.path.join(BASE_DIR, "dataset", "original")
DATASET_LBL_DIR = os.path.join(BASE_DIR, "dataset", "labels")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
REPORTS_IMG_DIR = os.path.join(REPORTS_DIR, "images")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(REPORTS_IMG_DIR, exist_ok=True)

class DatasetAnalyzer:
    def __init__(self):
        self.images = glob.glob(os.path.join(DATASET_IMG_DIR, "*.*"))
        self.images = [img for img in self.images if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.labels = glob.glob(os.path.join(DATASET_LBL_DIR, "*.txt"))
        
        self.stats = {
            "total_images": len(self.images),
            "total_labels_files": len(self.labels),
            "total_objects": 0,
            "classes": set(),
            "images_without_labels": 0,
            "labels_without_images": 0,
            "resolutions": [],
            "file_sizes": [],
            "bboxes": [],
            "per_image_stats": [],
            "brightness": [],
            "contrast": [],
            "sharpness": [],
            "entropy": [],
            "overexposed": [],
            "underexposed": [],
            "exif_data": []
        }
        
        self.rgb_hist = np.zeros((3, 256))
        self.hsv_hist = np.zeros((3, 256))
        self.heatmap = np.zeros((100, 100))

    def _get_exif_data(self, image_path):
        exif_info = {
            "fecha": "N/A", "hora": "N/A", "GPS": "N/A", "altitud": "N/A",
            "cámara": "N/A", "fabricante": "N/A", "modelo": "N/A", "orientación": "N/A"
        }
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag = TAGS.get(tag_id, tag_id)
                        if tag == 'DateTimeOriginal':
                            parts = str(value).split(' ')
                            if len(parts) == 2:
                                exif_info["fecha"] = parts[0]
                                exif_info["hora"] = parts[1]
                        elif tag == 'Make':
                            exif_info["fabricante"] = str(value).strip()
                        elif tag == 'Model':
                            exif_info["modelo"] = str(value).strip()
                        elif tag == 'Orientation':
                            exif_info["orientación"] = str(value)
                        elif tag == 'GPSInfo':
                            exif_info["GPS"] = "Presente"
        except Exception:
            pass
        return exif_info

    def _calc_entropy(self, gray_img):
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        logs = np.log2(hist + 1e-7)
        entropy = -1 * (hist * logs).sum()
        return entropy

    def analyze(self):
        console.print("[bold blue]Iniciando análisis del dataset...[/bold blue]")
        
        images_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in self.images}
        labels_basenames = {os.path.splitext(os.path.basename(p))[0]: p for p in self.labels}
        
        for name in labels_basenames:
            if name not in images_basenames:
                self.stats["labels_without_images"] += 1
                
        for img_path in track(self.images, description="Procesando imágenes..."):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            file_size = os.path.getsize(img_path)
            self.stats["file_sizes"].append(file_size)
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            h, w, _ = img.shape
            self.stats["resolutions"].append((w, h))
            
            # Calidad de imagen
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.stats["brightness"].append(float(np.mean(gray)))
            self.stats["contrast"].append(float(np.std(gray)))
            self.stats["sharpness"].append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            self.stats["entropy"].append(float(self._calc_entropy(gray)))
            
            total_pixels = w * h
            over_px = np.sum(gray > 240)
            under_px = np.sum(gray < 15)
            self.stats["overexposed"].append(float((over_px / total_pixels) * 100))
            self.stats["underexposed"].append(float((under_px / total_pixels) * 100))
            
            for i, col in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                self.rgb_hist[i] += hist.flatten()
                
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [256], [0, 256])
                self.hsv_hist[i] += hist.flatten()
            
            exif = self._get_exif_data(img_path)
            self.stats["exif_data"].append(exif)
            
            objects_count = 0
            occupied_area = 0
            
            if basename in labels_basenames:
                with open(labels_basenames[basename], 'r') as f:
                    lines = f.readlines()
                    objects_count = len(lines)
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            c, cx, cy, bw, bh = map(float, parts)
                            self.stats["classes"].add(int(c))
                            self.stats["bboxes"].append((cx, cy, bw, bh))
                            occupied_area += (bw * bh)
                            
                            x_idx = min(99, int(cx * 100))
                            y_idx = min(99, int(cy * 100))
                            self.heatmap[y_idx, x_idx] += 1
            else:
                self.stats["images_without_labels"] += 1
            
            self.stats["total_objects"] += objects_count
            
            self.stats["per_image_stats"].append({
                "nombre": os.path.basename(img_path),
                "resolucion": f"{w}x{h}",
                "tamaño_bytes": file_size,
                "cantidad_objetos": objects_count,
                "area_ocupada_pct": min(1.0, occupied_area) * 100,
                "nitidez": self.stats["sharpness"][-1],
                "entropia": self.stats["entropy"][-1],
                "brillo": self.stats["brightness"][-1]
            })

    def generate_figures(self):
        console.print("[bold cyan]Generando visualizaciones...[/bold cyan]")
        
        if self.stats["resolutions"]:
            areas = [w*h for w, h in self.stats["resolutions"]]
            plt.figure(figsize=(6, 4))
            plt.hist(areas, bins=20, color='skyblue', edgecolor='black')
            plt.title('Distribución de Resoluciones (Área)')
            plt.tight_layout()
            plt.savefig(os.path.join(REPORTS_IMG_DIR, 'resolutions_hist.png'))
            plt.close()
            
        plt.figure(figsize=(5, 5))
        plt.imshow(self.heatmap, cmap='hot', interpolation='nearest')
        plt.title('Heatmap Bounding Boxes')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_IMG_DIR, 'bbox_heatmap.png'))
        plt.close()
        
        if self.stats["bboxes"]:
            b_areas = [bw*bh for _, _, bw, bh in self.stats["bboxes"]]
            plt.figure(figsize=(6, 4))
            plt.hist(b_areas, bins=20, color='lightgreen', edgecolor='black')
            plt.title('Áreas de Bounding Boxes')
            plt.tight_layout()
            plt.savefig(os.path.join(REPORTS_IMG_DIR, 'bbox_areas_hist.png'))
            plt.close()
            
        plt.figure(figsize=(6, 4))
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            plt.plot(self.rgb_hist[i], color=col)
        plt.title('Histograma RGB Promedio')
        plt.xlim([0, 256])
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_IMG_DIR, 'rgb_hist.png'))
        plt.close()

    def _safe_avg(self, array):
        return float(sum(array) / len(array)) if array else 0.0

    def generate_reports(self):
        console.print("[bold green]Generando reportes finales y Fingerprint...[/bold green]")
        
        res_areas = [w*h for w, h in self.stats["resolutions"]]
        bboxes = self.stats["bboxes"]

        summary = {
            "general": {
                "total_imagenes": self.stats["total_images"],
                "total_archivos_etiquetas": self.stats["total_labels_files"],
                "total_objetos": self.stats["total_objects"],
                "clases_existentes": list(self.stats["classes"]),
                "imagenes_sin_etiqueta": self.stats["images_without_labels"],
                "etiquetas_sin_imagen": self.stats["labels_without_images"]
            },
            "calidad_imagen": {
                "brillo_promedio": self._safe_avg(self.stats["brightness"]),
                "contraste_promedio": self._safe_avg(self.stats["contrast"]),
                "nitidez_promedio": self._safe_avg(self.stats["sharpness"]),
                "entropia_promedio": self._safe_avg(self.stats["entropy"]),
                "sobreexposicion_promedio_pct": self._safe_avg(self.stats["overexposed"]),
                "subexposicion_promedio_pct": self._safe_avg(self.stats["underexposed"])
            },
            "bounding_boxes": {
                "cantidad_total": len(bboxes),
                "ancho_promedio": self._safe_avg([b[2] for b in bboxes]),
                "alto_promedio": self._safe_avg([b[3] for b in bboxes]),
                "area_promedio": self._safe_avg([b[2]*b[3] for b in bboxes])
            }
        }
        
        # Dataset Fingerprint
        fingerprint_str = json.dumps(summary, sort_keys=True).encode('utf-8')
        fingerprint_hash = hashlib.sha256(fingerprint_str).hexdigest()
        summary["fingerprint_hash"] = fingerprint_hash
        
        with open(os.path.join(REPORTS_DIR, 'dataset_fingerprint.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        with open(os.path.join(REPORTS_DIR, 'dataset_statistics.json'), 'w') as f:
            json.dump(summary, f, indent=4)
            
        # Export Excel (openpyxl) and CSV
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Estadisticas por Imagen"
        headers = ["nombre", "resolucion", "tamaño_bytes", "cantidad_objetos", "area_ocupada_pct", "nitidez", "entropia", "brillo"]
        ws.append(headers)
        
        with open(os.path.join(REPORTS_DIR, 'dataset_statistics.csv'), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for stat in self.stats["per_image_stats"]:
                writer.writerow(stat)
                ws.append([stat[h] for h in headers])
        
        wb.save(os.path.join(REPORTS_DIR, 'dataset_statistics.xlsx'))
            
        # Generar HTML
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reporte de Dataset ASFP</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
                h1, h2 { color: #2c3e50; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .grid { display: flex; flex-wrap: wrap; gap: 20px; }
                .card { border: 1px solid #ddd; padding: 10px; width: 45%; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Reporte de Análisis de Dataset</h1>
            <p><strong>Dataset Fingerprint:</strong> {{ data.fingerprint_hash }}</p>
            
            <h2>Resumen General</h2>
            <table>
                <tr><th>Total Imágenes</th><td>{{ data.general.total_imagenes }}</td></tr>
                <tr><th>Total Objetos</th><td>{{ data.general.total_objetos }}</td></tr>
            </table>
            
            <h2>Calidad de Imagen</h2>
            <table>
                <tr><th>Nitidez Promedio (Var. Laplaciano)</th><td>{{ data.calidad_imagen.nitidez_promedio | round(2) }}</td></tr>
                <tr><th>Entropía Promedio</th><td>{{ data.calidad_imagen.entropia_promedio | round(2) }}</td></tr>
                <tr><th>Sobreexposición %</th><td>{{ data.calidad_imagen.sobreexposicion_promedio_pct | round(2) }}</td></tr>
            </table>
            
            <h2>Visualizaciones</h2>
            <div class="grid">
                <div class="card">
                    <h3>Resoluciones</h3>
                    <img src="./images/resolutions_hist.png" alt="Hist Resoluciones">
                </div>
                <div class="card">
                    <h3>Spatial Heatmap</h3>
                    <img src="./images/bbox_heatmap.png" alt="Heatmap Bbox">
                </div>
                <div class="card">
                    <h3>Distribución RGB</h3>
                    <img src="./images/rgb_hist.png" alt="RGB Hist">
                </div>
            </div>
        </body>
        </html>
        """
        template = Template(html_template)
        html_content = template.render(data=summary)
        with open(os.path.join(REPORTS_DIR, 'dataset_report.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        # Generate PDF with reportlab
        self._generate_pdf(summary)
        console.print("[bold magenta]Análisis y reportes completados con éxito.[/bold magenta]")

    def _generate_pdf(self, summary):
        pdf_path = os.path.join(REPORTS_DIR, 'dataset_report.pdf')
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        elements.append(Paragraph("Reporte de Dataset ASFP", styles['Title']))
        elements.append(Paragraph(f"Fingerprint: {summary['fingerprint_hash']}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        data = [
            ["Métrica", "Valor"],
            ["Total Imágenes", str(summary['general']['total_imagenes'])],
            ["Total Objetos", str(summary['general']['total_objetos'])],
            ["Nitidez Promedio", f"{summary['calidad_imagen']['nitidez_promedio']:.2f}"],
            ["Entropía Promedio", f"{summary['calidad_imagen']['entropia_promedio']:.2f}"],
        ]
        
        t = Table(data, colWidths=[200, 100])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))
        
        # Add a couple of images if they exist
        img1 = os.path.join(REPORTS_IMG_DIR, 'bbox_heatmap.png')
        if os.path.exists(img1):
            elements.append(Paragraph("Spatial Heatmap", styles['Heading2']))
            elements.append(RLImage(img1, width=300, height=300))
            
        doc.build(elements)
