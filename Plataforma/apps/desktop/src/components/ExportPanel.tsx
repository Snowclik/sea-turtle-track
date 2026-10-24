import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { save } from '@tauri-apps/plugin-dialog';
import { useAnalysisStore } from '../store/analysisStore';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Download, FileSpreadsheet, FileCode, CheckCircle2 } from 'lucide-react';

export function ExportPanel() {
  const { result, videoName, imageName, status } = useAnalysisStore();
  const canExport = !!result && result.detections.length > 0;

  const currentMediaName = videoName || imageName;
  const baseName = currentMediaName
    ? currentMediaName.replace(/\.[^.]+$/, '')
    : 'detecciones';

  const exportFile = useCallback(async (format: 'csv' | 'json') => {
    if (!result) return;

    const ext      = format === 'csv' ? 'csv' : 'json';
    const filePath = await save({
      defaultPath: `${baseName}_tortugas.${ext}`,
      filters: [{ name: format.toUpperCase(), extensions: [ext] }],
    });

    if (!filePath) return;

    const payload = result.detections.map((d) => ({
      frameIndex:  d.frameIndex,
      timestamp:   d.timestamp,
      classId:     d.classId,
      className:   d.className,
      confidence:  d.confidence,
      bbox:        d.bbox,
    }));

    try {
      await invoke('export_results', {
        detections: payload,
        format,
        filePath,
      });
      alert(`Exportado exitosamente en: ${filePath}`);
    } catch (err) {
      alert(`Error al exportar: ${err}`);
    }
  }, [result, baseName]);

  if (status !== 'complete' && status !== 'cancelled') return null;

  return (
    <Card className="border-border shadow-clean">
      <CardHeader className="p-3.5 pb-2">
        <div className="flex items-center gap-2">
          <Download className="w-4 h-4 text-muted-foreground" />
          <CardTitle className="text-xs font-semibold">Exportar Detecciones</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="p-3.5 pt-1 space-y-2.5">
        {canExport ? (
          <>
            <div className="grid grid-cols-2 gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => exportFile('csv')}
                className="w-full text-xs gap-1.5"
              >
                <FileSpreadsheet className="w-3.5 h-3.5 text-emerald-600" />
                <span>CSV</span>
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={() => exportFile('json')}
                className="w-full text-xs gap-1.5"
              >
                <FileCode className="w-3.5 h-3.5 text-blue-600" />
                <span>JSON</span>
              </Button>
            </div>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              {result!.detections.length} registro{result!.detections.length !== 1 ? 's' : ''} con coordenadas, timestamp y clase.
            </p>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">
            No hay detecciones disponibles para exportar.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
