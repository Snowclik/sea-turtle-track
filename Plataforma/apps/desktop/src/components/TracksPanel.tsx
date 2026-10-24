import { useState } from 'react';
import { useAnalysisStore } from '../store/analysisStore';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { Eye, EyeOff, Layers, Scan, Timer } from 'lucide-react';

import { cn } from '@/lib/utils';

export function TracksPanel({ className }: { className?: string } = {}) {
  const { trackRegisters, selectedTrackId, setSelectedTrackId } = useAnalysisStore();
  const [previewTrackId, setPreviewTrackId] = useState<number | null>(null);
  const [showHitbox, setShowHitbox] = useState(true);

  const tracks = Object.values(trackRegisters).sort((a, b) => b.firstSeen - a.firstSeen);

  const formatTime = (s: number) => {
    const min = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`;
  };

  const previewTrack = previewTrackId !== null ? trackRegisters[previewTrackId] : null;

  return (
    <>
      <div className={cn("w-full lg:w-72 bg-background border-r-0 lg:border-r border-border flex flex-col h-full shrink-0 z-10 select-none pb-24 lg:pb-0", className)}>
        
        {/* Header */}
        <div className="h-12 px-4 border-b border-border flex items-center justify-between shrink-0 bg-background/50">
          <div className="flex items-center gap-2">
            <Scan className="w-3.5 h-3.5 text-muted-foreground" />
            <h2 className="text-xs font-semibold text-foreground tracking-tight">Rastros Detectados</h2>
          </div>
          <Badge variant="secondary" className="font-mono text-[10px] px-1.5 py-0">
            {tracks.length}
          </Badge>
        </div>

        {/* Empty state or list */}
        {tracks.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground p-6 text-center gap-2.5">
            <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center text-muted-foreground/60">
              <Layers className="w-5 h-5 stroke-1" />
            </div>
            <p className="text-xs font-medium text-foreground/80">Sin detecciones</p>
            <p className="text-[11px] text-muted-foreground leading-relaxed">
              Los rastros identificados por el modelo IA aparecerán listados aquí.
            </p>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {tracks.map((track) => {
              const isSelected = selectedTrackId === track.id;
              const duration = Math.max(0, track.lastSeen - track.firstSeen);

              return (
                <div
                  key={track.id}
                  onClick={() => {
                    setSelectedTrackId(isSelected ? null : track.id);
                    setPreviewTrackId(track.id);
                    setShowHitbox(true);
                  }}
                  className={`
                    group rounded-lg border p-2.5 transition-all duration-150 cursor-pointer
                    ${isSelected
                      ? 'border-primary ring-1 ring-primary bg-secondary/70 shadow-xs'
                      : 'border-border/80 hover:border-border hover:bg-secondary/40 bg-card'}
                  `}
                >
                  <div className="flex gap-2.5 items-center">
                    
                    {/* Thumbnail */}
                    <div className="w-14 h-14 bg-secondary rounded-md shrink-0 overflow-hidden border border-border/60 relative">
                      {track.thumbnailDataUrl ? (
                        <img
                          src={track.thumbnailDataUrl}
                          alt={`Track ${track.id}`}
                          className="w-full h-full object-cover transition-transform duration-200 group-hover:scale-105"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center text-muted-foreground/40">
                          <Scan className="w-4 h-4 stroke-1" />
                        </div>
                      )}
                    </div>

                    {/* Metadata */}
                    <div className="flex-1 min-w-0 flex flex-col justify-between py-0.5 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-xs font-bold text-foreground">
                          #{track.id.toString().padStart(3, '0')}
                        </span>
                        <Badge
                          variant={track.status === 'active' ? 'brand' : 'outline'}
                          className="text-[9px] px-1.5 py-0 uppercase tracking-wider font-semibold"
                        >
                          {track.status === 'active' ? 'Activo' : 'Cerrado'}
                        </Badge>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                        <span>Confianza:</span>
                        <span className="font-mono font-medium text-foreground">
                          {(track.maxConfidence * 100).toFixed(0)}%
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <Timer className="w-3 h-3" />
                          <span>{formatTime(track.firstSeen)}</span>
                        </span>
                        <span className="font-mono text-[10px] text-muted-foreground/80">
                          +{duration.toFixed(1)}s
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Preview Modal using Radix Dialog */}
      <Dialog open={previewTrack !== null} onOpenChange={(open) => !open && setPreviewTrackId(null)}>
        {previewTrack && (
          <DialogContent className="max-w-md">
            <DialogHeader>
              <div className="flex items-center gap-2">
                <DialogTitle className="text-sm font-semibold">
                  Rastro #{previewTrack.id.toString().padStart(3, '0')} · {previewTrack.className}
                </DialogTitle>
                <Badge
                  variant={previewTrack.status === 'active' ? 'brand' : 'outline'}
                  className="text-[10px]"
                >
                  {previewTrack.status}
                </Badge>
              </div>
              <DialogDescription className="text-xs text-muted-foreground">
                Recorte del cuadro con mayor nivel de confianza detectado por el modelo.
              </DialogDescription>
            </DialogHeader>

            <div className="flex flex-col items-center gap-4 py-2">
              
              {/* Crop & Hitbox canvas */}
              <div className="relative w-full aspect-square max-w-[280px] bg-secondary/80 rounded-lg overflow-hidden border border-border shadow-xs flex items-center justify-center">
                {previewTrack.thumbnailDataUrl ? (
                  <>
                    <img
                      src={previewTrack.thumbnailDataUrl}
                      alt={`Preview ${previewTrack.id}`}
                      className="w-full h-full object-contain"
                    />
                    {showHitbox && previewTrack.thumbnailBbox && (
                      <div
                        className="absolute border border-emerald-400 bg-emerald-500/15"
                        style={{
                          left: `${previewTrack.thumbnailBbox.x * 100}%`,
                          top: `${previewTrack.thumbnailBbox.y * 100}%`,
                          width: `${previewTrack.thumbnailBbox.w * 100}%`,
                          height: `${previewTrack.thumbnailBbox.h * 100}%`,
                        }}
                      >
                        <div className="absolute top-0 left-0 w-1.5 h-1.5 border-t border-l border-white" />
                        <div className="absolute top-0 right-0 w-1.5 h-1.5 border-t border-r border-white" />
                        <div className="absolute bottom-0 left-0 w-1.5 h-1.5 border-b border-l border-white" />
                        <div className="absolute bottom-0 right-0 w-1.5 h-1.5 border-b border-r border-white" />
                      </div>
                    )}
                  </>
                ) : (
                  <div className="text-xs text-muted-foreground">Sin captura disponible</div>
                )}
              </div>

              {/* Hitbox toggle button */}
              <div className="flex w-full justify-between items-center">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowHitbox(!showHitbox)}
                  className="text-xs"
                >
                  {showHitbox ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                  <span>{showHitbox ? 'Ocultar Cuadro Bounding' : 'Ver Cuadro Bounding'}</span>
                </Button>

                <span className="text-[11px] font-mono text-muted-foreground">
                  ID: {previewTrack.id}
                </span>
              </div>

              {/* Details table */}
              <div className="w-full grid grid-cols-2 gap-3 p-3 rounded-lg bg-secondary/50 border border-border/60 text-xs">
                <div>
                  <span className="block text-[10px] text-muted-foreground uppercase tracking-wider">
                    Confianza Máxima
                  </span>
                  <span className="font-mono text-sm font-semibold text-foreground">
                    {(previewTrack.maxConfidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div>
                  <span className="block text-[10px] text-muted-foreground uppercase tracking-wider">
                    Duración en Rastreo
                  </span>
                  <span className="font-mono text-sm font-semibold text-foreground">
                    {Math.max(0, previewTrack.lastSeen - previewTrack.firstSeen).toFixed(1)} s
                  </span>
                </div>
              </div>
            </div>
          </DialogContent>
        )}
      </Dialog>
    </>
  );
}
