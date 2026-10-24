import { useRef, useEffect, useCallback, useState, RefObject } from 'react';
import { useAnalysisStore } from '../store/analysisStore';
import { useVideoAnalysis } from '../hooks/useVideoAnalysis';
import { Video as VideoIcon, AlertCircle, Loader2, Radio } from 'lucide-react';

// Paleta de colores por track ID
const PALETTE = [
  '#22c55e', '#06b6d4', '#f59e0b', '#ef4444',
  '#a855f7', '#ec4899', '#14b8a6', '#f97316',
];
const trackColor = (id: number) => PALETTE[id % PALETTE.length];

interface VideoPlayerProps {
  videoRef: RefObject<HTMLVideoElement>;
}

export function VideoPlayer({ videoRef }: VideoPlayerProps) {
  const overlayRef   = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [videoLoading, setVideoLoading] = useState(false);
  const [videoError,   setVideoError]   = useState<string | null>(null);

  const { videoPath, tracks, status, progress, trackRegisters, selectedTrackId, setSelectedTrackId } = useAnalysisStore();
  const { startAnalysis, cancelAnalysis } = useVideoAnalysis(videoRef);

  // ── Sincronizar tamaño del canvas overlay ──────────────────
  const syncSize = useCallback(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    
    const video = videoRef.current;
    if (!video) return;
    if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
      canvas.width  = video.clientWidth;
      canvas.height = video.clientHeight;
    }
  }, [videoRef]);

  // ── Loop de dibujo (rAF propio del overlay) ─────────────────
  useEffect(() => {
    let rafId: number;

    const draw = () => {
      rafId = requestAnimationFrame(draw);
      const canvas = overlayRef.current;
      if (!canvas) return;
      
      if (!videoRef.current) return;

      syncSize();
      const W = canvas.width, H = canvas.height;
      let natW = W, natH = H;
      
      if (videoRef.current) {
        natW = videoRef.current.videoWidth;
        natH = videoRef.current.videoHeight;
      }
      
      if (natW === 0 || natH === 0) {
        natW = W; natH = H;
      }

      const scale = Math.min(W / natW, H / natH);
      const vidW = natW * scale;
      const vidH = natH * scale;
      const offsetX = (W - vidW) / 2;
      const offsetY = (H - vidH) / 2;

      const ctx = canvas.getContext('2d')!;
      ctx.clearRect(0, 0, W, H);

      const currentTracks = useAnalysisStore.getState().tracks;
      if (currentTracks.length === 0) return;

      for (const track of currentTracks) {
        const color = trackColor(track.id);
        const { x1, y1, x2, y2 } = track.bbox;

        const px = offsetX + x1 * vidW, py = offsetY + y1 * vidH;
        const pw = (x2 - x1) * vidW, ph = (y2 - y1) * vidH;

        // ── Trail (estela de movimiento) ─────────────────────
        if (track.trail.length >= 2) {
          ctx.beginPath();
          ctx.strokeStyle = color + '99';
          ctx.lineWidth   = 2;
          ctx.lineCap     = 'round';
          ctx.lineJoin    = 'round';

          const first = track.trail[0];
          ctx.moveTo(offsetX + first[0] * vidW, offsetY + first[1] * vidH);
          for (let i = 1; i < track.trail.length; i++) {
            const alpha = Math.round((i / track.trail.length) * 255).toString(16).padStart(2, '0');
            ctx.strokeStyle = color + alpha;
            ctx.lineTo(offsetX + track.trail[i][0] * vidW, offsetY + track.trail[i][1] * vidH);
          }
          ctx.stroke();

          // Punto en la posición actual
          const last = track.trail[track.trail.length - 1];
          ctx.beginPath();
          ctx.arc(offsetX + last[0] * vidW, offsetY + last[1] * vidH, 4, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }

        // ── Bounding box ─────────────────────────────────────
        ctx.strokeStyle = color;
        ctx.lineWidth   = 2;
        ctx.strokeRect(px, py, pw, ph);

        // Esquinas decorativas de precisión
        const cs = Math.min(pw, ph) * 0.18;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(px, py + cs); ctx.lineTo(px, py); ctx.lineTo(px + cs, py);
        ctx.moveTo(px + pw - cs, py); ctx.lineTo(px + pw, py); ctx.lineTo(px + pw, py + cs);
        ctx.moveTo(px + pw, py + ph - cs); ctx.lineTo(px + pw, py + ph); ctx.lineTo(px + pw - cs, py + ph);
        ctx.moveTo(px + cs, py + ph); ctx.lineTo(px, py + ph); ctx.lineTo(px, py + ph - cs);
        ctx.stroke();

        // ── Etiqueta ─────────────────────────────────────────
        const confPct = (track.confidence * 100).toFixed(0);
        const label   = `#${track.id} ${track.className} ${confPct}%`;
        ctx.font = '500 11px "Geist Variable", system-ui, sans-serif';
        const textW = ctx.measureText(label).width + 12;
        const labelY = py > 24 ? py - 6 : py + ph + 18;

        ctx.fillStyle = 'rgba(15, 15, 18, 0.85)';
        ctx.beginPath();
        ctx.roundRect(px, labelY - 16, textW, 18, 4);
        ctx.fill();

        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(px, labelY - 16, textW, 18, 4);
        ctx.stroke();

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, px + 6, labelY - 3);

        // ── Highlight si está seleccionado ──
        if (selectedTrackId === track.id) {
          ctx.fillStyle = color + '22';
          ctx.fillRect(px, py, pw, ph);
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(px - 2, py - 2, pw + 4, ph + 4);
        } else if (selectedTrackId !== null) {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
          ctx.fillRect(px - 2, py - 2, pw + 4, ph + 4);
        }
      }
    };

    rafId = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafId);
  }, [syncSize, videoRef, selectedTrackId]);

  // ── Manejador de clics en BBoxes ──
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    const currentTracks = useAnalysisStore.getState().tracks;
    if (!canvas || !video || currentTracks.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const W = canvas.width, H = canvas.height;
    let natW = video.videoWidth || W;
    let natH = video.videoHeight || H;

    const scale = Math.min(W / natW, H / natH);
    const vidW = natW * scale;
    const vidH = natH * scale;
    const offsetX = (W - vidW) / 2;
    const offsetY = (H - vidH) / 2;

    let clickedTrackId: number | null = null;
    let minArea = Infinity;

    for (const track of currentTracks) {
      const px = offsetX + track.bbox.x1 * vidW;
      const py = offsetY + track.bbox.y1 * vidH;
      const pw = (track.bbox.x2 - track.bbox.x1) * vidW;
      const ph = (track.bbox.y2 - track.bbox.y1) * vidH;

      if (clickX >= px && clickX <= px + pw && clickY >= py && clickY <= py + ph) {
        const area = pw * ph;
        if (area < minArea) {
          minArea = area;
          clickedTrackId = track.id;
        }
      }
    }

    if (clickedTrackId !== null) {
      setSelectedTrackId(clickedTrackId === selectedTrackId ? null : clickedTrackId);
    } else {
      setSelectedTrackId(null);
    }
  };

  useEffect(() => {
    if (!videoPath) return;
    if (videoRef.current && videoRef.current.src !== videoPath) {
      videoRef.current.src = videoPath;
      videoRef.current.load();
    }
  }, [videoPath, videoRef]);

  useEffect(() => {
    setVideoError(null);
    setVideoLoading(!!videoPath);
  }, [videoPath]);

  const analysisPct = progress
    ? Math.min(100, (progress.currentTimestamp / Math.max(progress.videoDuration, 1)) * 100)
    : 0;

  const isAnalyzing  = status === 'analyzing';
  const isCancelling = status === 'cancelling';

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-video bg-neutral-950 rounded-xl overflow-hidden select-none flex flex-col justify-center border border-border shadow-clean"
    >
      {/* Video */}
      <video
        ref={videoRef}
        src={videoPath ?? undefined}
        className="w-full h-full object-contain"
        crossOrigin="anonymous"
        preload="metadata"
        onLoadStart={() => { setVideoLoading(true); setVideoError(null); }}
        onLoadedMetadata={() => {
          setVideoLoading(false);
          syncSize();
          const dur = videoRef.current?.duration ?? 0;
          useAnalysisStore.getState().setVideoDuration(dur);
        }}
        onCanPlay={() => setVideoLoading(false)}
        onError={(e) => {
          const code = e.currentTarget.error?.code ?? -1;
          const msgs: Record<number, string> = {
            1: 'Carga abortada', 2: 'Error de red',
            3: 'Formato no soportado', 4: 'Fuente inaccesible',
          };
          setVideoLoading(false);
          setVideoError(msgs[code] ?? `Error (código ${code})`);
        }}
      />

      {/* Canvas overlay — tracks en tiempo real */}
      <canvas
        ref={overlayRef}
        onClick={handleCanvasClick}
        className="absolute inset-0 cursor-crosshair"
        style={{ width: '100%', height: '100%' }}
      />

      {/* Empty State */}
      {!videoPath && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-3 p-6 text-center">
          <div className="w-12 h-12 rounded-xl bg-neutral-900 border border-neutral-800 flex items-center justify-center text-neutral-400 shadow-inner">
            <VideoIcon className="w-6 h-6 stroke-1 text-neutral-400" />
          </div>
          <div className="space-y-1">
            <p className="text-sm font-medium text-neutral-200">Visor de Video UAV</p>
            <p className="text-xs text-neutral-500 max-w-sm">
              Carga una grabación de dron desde la barra inferior para ejecutar la inferencia y registrar rastros.
            </p>
          </div>
        </div>
      )}

      {/* Loading Video Overlay */}
      {videoPath && videoLoading && !videoError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-xs gap-2.5 text-white">
          <Loader2 className="w-6 h-6 animate-spin text-neutral-300" />
          <p className="text-xs font-medium text-neutral-300">Cargando flujo de video...</p>
        </div>
      )}

      {/* Error Overlay */}
      {videoError && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/75 backdrop-blur-xs gap-2.5 px-6 text-center text-white">
          <AlertCircle className="w-8 h-8 text-destructive" />
          <p className="text-sm font-semibold text-white">{videoError}</p>
          <p className="text-xs text-neutral-400">Verifica el códec del archivo (MP4 / WebM compatible)</p>
        </div>
      )}

      {/* Model Loading Status Badge */}
      {status === 'loading_model' && (
        <div className="absolute top-3.5 left-3.5 flex items-center gap-2 bg-neutral-900/90 border border-neutral-700/80 rounded-full px-3 py-1 shadow-sm backdrop-blur-md">
          <Loader2 className="w-3 h-3 text-amber-400 animate-spin" />
          <span className="text-[11px] font-mono text-neutral-200">Cargando pesos ONNX...</span>
        </div>
      )}

      {/* Analyzing HUD Badge */}
      {(isAnalyzing || isCancelling) && (
        <div className="absolute top-3.5 left-3.5 flex items-center gap-2 bg-neutral-900/90 border border-neutral-700/80 rounded-full px-3 py-1 shadow-sm backdrop-blur-md">
          <span className={`w-2 h-2 rounded-full ${isCancelling ? 'bg-destructive' : 'bg-brand animate-pulse'}`} />
          <span className="text-[11px] font-mono text-neutral-200 font-medium">
            {isCancelling ? 'Cancelando...' : `Inferencia · ${analysisPct.toFixed(0)}%`}
          </span>
        </div>
      )}

      {/* Active Tracks Count Badge */}
      {isAnalyzing && tracks.length > 0 && (
        <div className="absolute top-3.5 right-3.5 flex items-center gap-1.5 bg-neutral-900/90 border border-neutral-700/80 rounded-full px-2.5 py-1 shadow-sm backdrop-blur-md">
          <Radio className="w-3 h-3 text-brand" />
          <span className="text-[11px] font-mono text-neutral-200 font-medium">
            {tracks.length} activo{tracks.length !== 1 ? 's' : ''}
          </span>
        </div>
      )}

      {/* Bottom Progress Bar */}
      {(isAnalyzing || isCancelling) && analysisPct > 0 && (
        <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-neutral-800">
          <div
            className={`h-full transition-all duration-300 ${isCancelling ? 'bg-destructive' : 'bg-primary'}`}
            style={{ width: `${analysisPct}%` }}
          />
        </div>
      )}
    </div>
  );
}
