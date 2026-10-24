import { useState, useCallback, useEffect, RefObject } from 'react';
import { open } from '@tauri-apps/plugin-dialog';
import { convertFileSrc } from '@tauri-apps/api/core';
import { useAnalysisStore } from '../store/analysisStore';
import { useVideoAnalysis } from '../hooks/useVideoAnalysis';
import { useImageAnalysis } from '../hooks/useImageAnalysis';
import { Button } from '@/components/ui/button';
import {
  Video,
  Image as ImageIcon,
  RotateCcw,
  RotateCw,
  Play,
  Pause,
  Scan,
  Square,
  SlidersHorizontal,
} from 'lucide-react';

interface ControlBarProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  imageRef: RefObject<HTMLImageElement | null>;
}

export function ControlBar({ videoRef, imageRef }: ControlBarProps) {
  const [isPlaying,   setIsPlaying]   = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  const { status, videoDuration, videoName, imageName, mediaType, config, setConfig, progress } =
    useAnalysisStore();

  const { startAnalysis: startVideoAnalysis, cancelAnalysis: cancelVideoAnalysis } = useVideoAnalysis(videoRef);
  const { startAnalysis: startImageAnalysis, cancelAnalysis: cancelImageAnalysis } = useImageAnalysis(imageRef);

  const isAnalyzing  = (status === 'analyzing' || status === 'loading_model' || status === 'cancelling');
  const isCancelling = status === 'cancelling';

  // ── Eventos del elemento video ──────────────────────────────
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTimeUpdate = () => setCurrentTime(v.currentTime);
    const onEnded      = () => setIsPlaying(false);
    const onPause      = () => setIsPlaying(false);
    const onPlay       = () => setIsPlaying(true);
    v.addEventListener('timeupdate', onTimeUpdate);
    v.addEventListener('ended',      onEnded);
    v.addEventListener('pause',      onPause);
    v.addEventListener('play',       onPlay);
    return () => {
      v.removeEventListener('timeupdate', onTimeUpdate);
      v.removeEventListener('ended',      onEnded);
      v.removeEventListener('pause',      onPause);
      v.removeEventListener('play',       onPlay);
    };
  }, [videoRef, videoDuration]);

  // ── Sincronizar scrubber con el frame analizado ─────────────
  useEffect(() => {
    if (status === 'analyzing' && progress) {
      setCurrentTime(progress.currentTimestamp);
    }
  }, [status, progress]);

  // ── Seleccionar video ───────────────────────────────────────
  const handleSelectVideo = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: 'Videos', extensions: ['mp4', 'mov', 'avi', 'mkv', 'webm'] }],
    });

    if (!selected || typeof selected !== 'string') return;

    const assetUrl = convertFileSrc(selected);
    const store = useAnalysisStore.getState();
    store.resetAnalysis();
    store.setVideoPath(assetUrl, selected.split(/[\\/]/).pop() ?? selected);

    if (videoRef.current) {
      videoRef.current.src = assetUrl;
      videoRef.current.load();
    }
    
    setCurrentTime(0);
    setIsPlaying(false);
  }, [videoRef]);

  const handleSelectImage = useCallback(async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: 'Imágenes', extensions: ['jpg', 'jpeg', 'png', 'webp'] }],
    });

    if (!selected || typeof selected !== 'string') return;

    const assetUrl = convertFileSrc(selected);
    const store = useAnalysisStore.getState();
    store.resetAnalysis();
    store.setImagePath(assetUrl, selected.split(/[\\/]/).pop() ?? selected);
  }, []);

  // ── Controles de reproducción ───────────────────────────────
  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) { v.play(); setIsPlaying(true); }
    else          { v.pause(); setIsPlaying(false); }
  }, [videoRef]);

  const seek = useCallback((delta: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(v.duration, v.currentTime + delta));
  }, [videoRef]);

  const handleScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const v = videoRef.current;
    if (!v) return;
    const t = parseFloat(e.target.value);
    v.currentTime = t;
    setCurrentTime(t);
  }, [videoRef]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60);
    return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  };

  const analysisPct = progress
    ? Math.min(100, (progress.currentFrame / progress.totalFrames) * 100)
    : 0;

  return (
    <div className="bg-card border border-border rounded-xl p-3.5 shadow-clean flex flex-col gap-3">

      {/* Scrubber de tiempo */}
      {mediaType === 'video' && videoDuration > 0 && (
        <div className="flex items-center gap-3 px-1">
          <span className="text-[11px] text-muted-foreground font-mono w-10 shrink-0">
            {formatTime(currentTime)}
          </span>
          <div className="relative flex-1 flex items-center">
            <input
              type="range"
              min={0}
              max={videoDuration}
              step={0.1}
              value={currentTime}
              onChange={handleScrub}
              disabled={isAnalyzing}
              className="w-full accent-primary h-1.5 rounded-full cursor-pointer disabled:cursor-default"
            />
            {isAnalyzing && analysisPct > 0 && (
              <div
                className="absolute top-1/2 -translate-y-1/2 left-0 h-1.5 bg-brand/30 rounded-full pointer-events-none"
                style={{ width: `${analysisPct}%` }}
              />
            )}
          </div>
          <span className="text-[11px] text-muted-foreground font-mono w-10 shrink-0 text-right">
            {formatTime(videoDuration)}
          </span>
        </div>
      )}

      {/* Controles adaptables */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
        
        {/* Medios y reproducción */}
        <div className="flex items-center justify-between sm:justify-start gap-2 flex-wrap">
          <div className="flex items-center gap-1.5">
            <Button
              variant="outline"
              size="sm"
              onClick={handleSelectVideo}
              disabled={isAnalyzing}
              className="h-7 px-2.5 text-xs"
            >
              <Video className="w-3.5 h-3.5" />
              <span>Video</span>
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={handleSelectImage}
              disabled={isAnalyzing}
              className="h-7 px-2.5 text-xs"
            >
              <ImageIcon className="w-3.5 h-3.5" />
              <span>Imagen</span>
            </Button>
          </div>

          {/* Player controls */}
          {mediaType === 'video' && (
            <div className="flex items-center gap-1 bg-secondary/70 p-0.5 rounded-lg border border-border/50">
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={() => seek(-10)}
                disabled={!videoDuration || isAnalyzing}
                title="Retroceder 10s"
              >
                <RotateCcw className="w-3 h-3" />
              </Button>

              <Button
                variant={isPlaying ? "outline" : "default"}
                size="icon-xs"
                onClick={togglePlay}
                disabled={!videoDuration || isAnalyzing}
                title={isPlaying ? 'Pausar' : 'Reproducir'}
              >
                {isPlaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3 ml-0.5" />}
              </Button>

              <Button
                variant="ghost"
                size="icon-xs"
                onClick={() => seek(10)}
                disabled={!videoDuration || isAnalyzing}
                title="Avanzar 10s"
              >
                <RotateCw className="w-3 h-3" />
              </Button>
            </div>
          )}
        </div>

        {/* Parámetros y Acción */}
        <div className="flex items-center justify-between sm:justify-end gap-2.5 flex-wrap pt-1.5 sm:pt-0 border-t border-border/40 sm:border-t-0">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
              <span>FPS:</span>
              <select
                value={config.analysisFps}
                onChange={(e) => setConfig({ analysisFps: parseInt(e.target.value) })}
                disabled={isAnalyzing}
                className="h-7 text-xs bg-background border border-border rounded-md px-1.5 text-foreground outline-none focus:ring-1 focus:ring-ring cursor-pointer"
              >
                {[2, 3, 5, 8, 10].map((fps) => (
                  <option key={fps} value={fps}>{fps}</option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
              <span>Conf:</span>
              <select
                value={config.confidenceThreshold}
                onChange={(e) => setConfig({ confidenceThreshold: parseFloat(e.target.value) })}
                disabled={isAnalyzing}
                className="h-7 text-xs bg-background border border-border rounded-md px-1.5 text-foreground outline-none focus:ring-1 focus:ring-ring cursor-pointer"
              >
                {[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50].map((v) => (
                  <option key={v} value={v}>{(v * 100).toFixed(0)}%</option>
                ))}
              </select>
            </div>
          </div>

          {/* Action button */}
          {isAnalyzing ? (
            <Button
              variant="destructive"
              size="sm"
              onClick={() => mediaType === 'video' ? cancelVideoAnalysis() : cancelImageAnalysis()}
              disabled={isCancelling}
              className="h-7 px-3 text-xs"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>{isCancelling ? '...' : 'Detener'}</span>
            </Button>
          ) : (
            <Button
              variant="default"
              size="sm"
              onClick={() => mediaType === 'video' ? startVideoAnalysis() : startImageAnalysis()}
              disabled={!mediaType}
              className="h-7 px-3 text-xs"
            >
              <Scan className="w-3.5 h-3.5" />
              <span>Analizar</span>
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
