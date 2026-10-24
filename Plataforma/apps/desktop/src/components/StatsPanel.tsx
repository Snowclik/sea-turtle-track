import { useAnalysisStore } from '../store/analysisStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ExportPanel } from './ExportPanel';
import {
  Activity,
  CheckCircle2,
  Clock,
  Cpu,
  Layers,
  Sparkles,
  AlertCircle,
  Loader2,
} from 'lucide-react';

export function StatsPanel() {
  const { status, progress, result, trackRegisters, hardware, error } = useAnalysisStore();

  const tracks = Object.values(trackRegisters);
  const activeTracks = tracks.filter(t => t.status === 'active').length;
  const finishedTracks = tracks.filter(t => t.status === 'finished').length;
  const avgDuration = tracks.length > 0 ? tracks.reduce((sum, t) => sum + Math.max(0, t.lastSeen - t.firstSeen), 0) / tracks.length : 0;
  const avgConf = tracks.length > 0 ? tracks.reduce((sum, t) => sum + t.maxConfidence, 0) / tracks.length : 0;

  const formatMs  = (ms: number) => ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(1)} s`;
  const formatTime = (s: number) => {
    if (s < 60) return `${s.toFixed(0)} s`;
    return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  };
  const pct = (p: any) => {
    if (!p) return 0;
    return Math.min(100, (p.currentFrame / p.totalFrames) * 100);
  };

  return (
    <div className="flex flex-col gap-3.5">

      {/* Hardware info */}
      {hardware && (
        <Card className="shadow-clean border-border">
          <CardContent className="p-3.5">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                Motor de Inferencia
              </span>
              <Badge variant="outline" className="text-[10px] px-1.5 py-0">
                ONNX Web
              </Badge>
            </div>
            <p className="text-xs font-semibold text-foreground flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-muted-foreground" />
              {hardware.executionProvider}
            </p>
            <p className="text-[11px] text-muted-foreground mt-1">
              Clases activas: <span className="text-foreground/80 font-medium">{hardware.classNames.join(', ')}</span>
            </p>
          </CardContent>
        </Card>
      )}

      {/* Model loading state */}
      {status === 'loading_model' && (
        <Card className="border-amber-200/60 bg-amber-50/50 shadow-clean">
          <CardContent className="p-3.5 flex items-center gap-3">
            <Loader2 className="w-4 h-4 text-amber-600 animate-spin shrink-0" />
            <div>
              <p className="text-xs font-medium text-amber-900">Cargando pesos ONNX...</p>
              <p className="text-[11px] text-amber-700/80 mt-0.5">Inicializando buffers WebAssembly</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* In progress analysis */}
      {(status === 'analyzing' || status === 'cancelling') && progress && (
        <Card className="shadow-clean border-border">
          <CardHeader className="p-3.5 pb-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-brand animate-ping" />
                <CardTitle className="text-xs font-semibold">
                  {status === 'cancelling' ? 'Cancelando...' : 'Analizando Stream'}
                </CardTitle>
              </div>
              <span className="font-mono text-xs font-bold text-foreground">
                {pct(progress).toFixed(0)}%
              </span>
            </div>
          </CardHeader>
          <CardContent className="p-3.5 pt-1 space-y-3">
            {/* Smooth progress bar */}
            <div className="w-full h-1.5 bg-secondary rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-200 ${status === 'cancelling' ? 'bg-destructive' : 'bg-primary'}`}
                style={{ width: `${pct(progress)}%` }}
              />
            </div>

            <div className="grid grid-cols-2 gap-2.5 pt-1">
              <StatMetric label="Rastros Totales" value={String(tracks.length)} />
              <StatMetric label="En Cuadro" value={String(activeTracks)} highlight />
              <StatMetric label="Finalizados" value={String(finishedTracks)} />
              <StatMetric label="Confianza Media" value={`${(avgConf * 100).toFixed(0)}%`} />
              <StatMetric label="Duración Media" value={`${avgDuration.toFixed(1)}s`} />
              <StatMetric label="Tiempo Video" value={`${progress.currentTimestamp.toFixed(1)}s`} />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Completed results */}
      {result && (
        <Card className="border-border shadow-clean">
          <CardHeader className="p-3.5 pb-2">
            <div className="flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-600" />
              <CardTitle className="text-xs font-semibold text-foreground">
                Análisis Finalizado
              </CardTitle>
            </div>
          </CardHeader>

          <CardContent className="p-3.5 pt-1 space-y-3.5">
            <div className="grid grid-cols-2 gap-2">
              <div className="bg-secondary/60 rounded-lg p-2.5 text-center border border-border/50">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground block font-medium">Total Rastros</span>
                <span className="text-xl font-bold font-mono text-foreground">{tracks.length}</span>
              </div>
              <div className="bg-secondary/60 rounded-lg p-2.5 text-center border border-border/50">
                <span className="text-[10px] uppercase tracking-wide text-muted-foreground block font-medium">FPS Rendimiento</span>
                <span className="text-xl font-bold font-mono text-foreground">{result.realFps.toFixed(1)}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <StatMetric label="Tiempo Inferencia" value={formatMs(result.processingTimeMs)} />
              <StatMetric label="Duración Video" value={formatTime(result.videoDuration)} />
              <StatMetric label="Confianza Media" value={`${(avgConf * 100).toFixed(1)}%`} />
              <StatMetric label="Duración Rastro" value={`${avgDuration.toFixed(1)}s`} />
            </div>

            {/* Distribution */}
            {tracks.length > 0 && (
              <ClassBreakdown tracks={tracks} />
            )}
          </CardContent>
        </Card>
      )}

      {/* Idle placeholder */}
      {status === 'idle' && !result && (
        <div className="rounded-xl border border-dashed border-border p-6 text-center text-muted-foreground space-y-1">
          <Layers className="w-8 h-8 mx-auto text-muted-foreground/40 stroke-1" />
          <p className="text-xs font-medium text-foreground/80 mt-2">Panel de Métricas</p>
          <p className="text-[11px] text-muted-foreground">
            Carga un archivo y haz clic en Iniciar Análisis para ver telemetría.
          </p>
        </div>
      )}

      {/* Error notification */}
      {status === 'error' && (
        <Card className="border-destructive/30 bg-destructive/5 shadow-clean">
          <CardContent className="p-3.5 flex items-start gap-2.5 text-xs text-destructive">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold">Error de Inferencia</p>
              {error && <p className="font-mono text-[10px] opacity-90 mt-1">{error}</p>}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Export Panel Integration */}
      <ExportPanel />
    </div>
  );
}

function StatMetric({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] text-muted-foreground uppercase tracking-wider">{label}</span>
      <span className={`font-mono text-xs font-semibold ${highlight ? 'text-brand' : 'text-foreground'}`}>
        {value}
      </span>
    </div>
  );
}

function ClassBreakdown({ tracks }: { tracks: { className: string }[] }) {
  const counts = tracks.reduce<Record<string, number>>((acc, t) => {
    acc[t.className] = (acc[t.className] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="pt-2 border-t border-border">
      <span className="text-[10px] text-muted-foreground uppercase tracking-wider block mb-2 font-medium">
        Distribución por Clase
      </span>
      <div className="space-y-1.5">
        {Object.entries(counts).map(([cls, count]) => (
          <div key={cls} className="flex items-center justify-between text-xs py-0.5 px-2 rounded-md bg-secondary/40">
            <div className="flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-foreground/60" />
              <span className="text-foreground/80 font-medium">{cls}</span>
            </div>
            <span className="font-mono text-xs font-semibold text-foreground">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
