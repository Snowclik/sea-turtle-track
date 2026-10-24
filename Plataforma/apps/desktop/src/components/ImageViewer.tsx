import { useRef, useEffect, useCallback, useState, RefObject } from 'react';
import { useAnalysisStore } from '../store/analysisStore';
import { Image as ImageIcon, Loader2 } from 'lucide-react';

const PALETTE = [
  '#22c55e', '#06b6d4', '#f59e0b', '#ef4444',
  '#a855f7', '#ec4899', '#14b8a6', '#f97316',
];
const trackColor = (id: number) => PALETTE[id % PALETTE.length];

interface ImageViewerProps {
  imageRef: RefObject<HTMLImageElement | null>;
}

export function ImageViewer({ imageRef }: ImageViewerProps) {
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [imageLoading, setImageLoading] = useState(false);

  const { imagePath, tracks, status, selectedTrackId, setSelectedTrackId } = useAnalysisStore();

  const syncSize = useCallback(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    
    const img = imageRef.current;
    if (!img) return;
    if (canvas.width !== img.clientWidth || canvas.height !== img.clientHeight) {
      canvas.width = img.clientWidth;
      canvas.height = img.clientHeight;
    }
  }, [imageRef]);

  // Dibujar detecciones en la imagen
  useEffect(() => {
    let rafId: number;

    const draw = () => {
      rafId = requestAnimationFrame(draw);
      const canvas = overlayRef.current;
      if (!canvas) return;
      
      if (!imageRef.current) return;

      syncSize();
      const W = canvas.width, H = canvas.height;
      let natW = W, natH = H;
      
      if (imageRef.current) {
        natW = imageRef.current.naturalWidth;
        natH = imageRef.current.naturalHeight;
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

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(px, py, pw, ph);

        const cs = Math.min(pw, ph) * 0.18;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(px, py + cs); ctx.lineTo(px, py); ctx.lineTo(px + cs, py);
        ctx.moveTo(px + pw - cs, py); ctx.lineTo(px + pw, py); ctx.lineTo(px + pw, py + cs);
        ctx.moveTo(px + pw, py + ph - cs); ctx.lineTo(px + pw, py + ph); ctx.lineTo(px + pw - cs, py + ph);
        ctx.moveTo(px + cs, py + ph); ctx.lineTo(px, py + ph); ctx.lineTo(px, py + ph - cs);
        ctx.stroke();

        const confPct = (track.confidence * 100).toFixed(0);
        const label = `#${track.id} ${track.className} ${confPct}%`;
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

        ctx.fillStyle = '#fff';
        ctx.fillText(label, px + 6, labelY - 3);
        
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
  }, [syncSize, imageRef, selectedTrackId]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = overlayRef.current;
    const img = imageRef.current;
    const currentTracks = useAnalysisStore.getState().tracks;
    if (!canvas || !img || currentTracks.length === 0) return;

    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const W = canvas.width, H = canvas.height;
    const natW = img.naturalWidth || W;
    const natH = img.naturalHeight || H;
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

  const isAnalyzing = status === 'analyzing';

  return (
    <div
      ref={containerRef}
      className="relative w-full aspect-video bg-neutral-950 rounded-xl overflow-hidden select-none flex flex-col justify-center border border-border shadow-clean"
    >
      <img
        ref={imageRef as React.RefObject<HTMLImageElement>}
        src={imagePath ?? undefined}
        className="w-full h-full object-contain"
        crossOrigin="anonymous"
        onLoad={() => { setImageLoading(false); syncSize(); }}
        onError={() => { setImageLoading(false); }}
      />

      <canvas
        ref={overlayRef}
        onClick={handleCanvasClick}
        className="absolute inset-0 cursor-crosshair"
        style={{ width: '100%', height: '100%' }}
      />

      {!imagePath && (
        <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground gap-3 p-6 text-center">
          <div className="w-12 h-12 rounded-xl bg-neutral-900 border border-neutral-800 flex items-center justify-center text-neutral-400 shadow-inner">
            <ImageIcon className="w-6 h-6 stroke-1 text-neutral-400" />
          </div>
          <div className="space-y-1">
            <p className="text-sm font-medium text-neutral-200">Visor de Imagen Aérea</p>
            <p className="text-xs text-neutral-500 max-w-sm">
              Selecciona una fotografía ortomosaico o captura para ejecutar la inferencia de rastros.
            </p>
          </div>
        </div>
      )}

      {imagePath && imageLoading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-xs gap-2.5 text-white">
          <Loader2 className="w-6 h-6 animate-spin text-neutral-300" />
          <p className="text-xs font-medium text-neutral-300">Cargando imagen...</p>
        </div>
      )}

      {status === 'loading_model' && (
        <div className="absolute top-3.5 left-3.5 flex items-center gap-2 bg-neutral-900/90 border border-neutral-700/80 rounded-full px-3 py-1 shadow-sm backdrop-blur-md">
          <Loader2 className="w-3 h-3 text-amber-400 animate-spin" />
          <span className="text-[11px] font-mono text-neutral-200">Cargando modelo...</span>
        </div>
      )}

      {isAnalyzing && (
        <div className="absolute top-3.5 left-3.5 flex items-center gap-2 bg-neutral-900/90 border border-neutral-700/80 rounded-full px-3 py-1 shadow-sm backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-brand animate-pulse" />
          <span className="text-[11px] font-mono text-neutral-200 font-medium">Analizando imagen...</span>
        </div>
      )}

      {status === 'complete' && tracks.length > 0 && (
        <div className="absolute top-3.5 right-3.5 flex items-center gap-1.5 bg-neutral-900/90 border border-neutral-700/80 rounded-full px-2.5 py-1 shadow-sm backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-brand" />
          <span className="text-[11px] font-mono text-neutral-200 font-medium">
            {tracks.length} detección{tracks.length !== 1 ? 'es' : ''}
          </span>
        </div>
      )}
    </div>
  );
}
