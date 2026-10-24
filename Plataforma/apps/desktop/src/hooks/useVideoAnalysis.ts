import { useRef, useCallback, RefObject } from 'react';
import * as onnxService from '../services/onnxService';
import { IoUTracker } from '../services/tracker';
import type { Detection } from '../types';
import { useAnalysisStore } from '../store/analysisStore';
import type { ModelId } from '../store/analysisStore';

const CAPTURE_SIZE    = 640;

export function useVideoAnalysis(
  videoRef: RefObject<HTMLVideoElement | null>
) {
  const cancelledRef = useRef(false);
  const rafRef       = useRef<number>(0);
  const canvas       = useRef<HTMLCanvasElement | null>(null);
  const tracker      = useRef(new IoUTracker());
  const lastInfTs    = useRef(0);
  const inferring    = useRef(false);

  const {
    config, setStatus, setProgress,
    addDetection, setResult, setError, setHardware,
    resetAnalysis, setTracks,
    activeModelId, trackRegisters, upsertTrackRegister, markMissingTracksAsFinished,
  } = useAnalysisStore();

  // ── Carga del modelo (una sola vez) ──────────────────────────
  const ensureModel = useCallback(async (modelId: ModelId) => {
    if (onnxService.isModelLoaded(modelId)) return;
    setStatus('loading_model');
    const { classNames, numClasses } = await onnxService.loadModel(modelId);
    setHardware({
      executionProvider: 'WASM (onnxruntime-web)',
      modelLoaded: true,
      classNames,
      numClasses,
    });
  }, [setStatus, setHardware]);

  // ── Análisis en tiempo real ──────────────────────────────────
  const startAnalysis = useCallback(async () => {
    console.info(`[TRACKER] [${new Date().toLocaleTimeString()}] ▶️ Iniciando análisis de video`);
    const video = videoRef.current;
    if (!video?.src) {
      setError('Selecciona un video antes de analizar.');
      setStatus('error');
      return;
    }

    resetAnalysis();
    tracker.current.reset();
    cancelledRef.current = false;

    try {
      const currentModelId = useAnalysisStore.getState().activeModelId;
      await ensureModel(currentModelId);
      if (cancelledRef.current) { setStatus('cancelled'); return; }

      // Canvas offscreen reutilizable
      if (!canvas.current) {
        canvas.current = document.createElement('canvas');
        canvas.current.width = canvas.current.height = CAPTURE_SIZE;
      }
      const cvs = canvas.current;
      const ctx = cvs.getContext('2d', { willReadFrequently: true })!;

      const allDetections: Detection[] = [];
      const tStart        = performance.now();
      const minIntervalMs = 1000 / config.analysisFps;

      // Rebobinar y reproducir
      video.currentTime = 0;
      await new Promise<void>((res) => {
        if (video.readyState >= 2) { res(); return; }
        video.addEventListener('canplay', () => res(), { once: true });
        setTimeout(res, 3000);
      });

      setStatus('analyzing');
      video.play();

      // ── Loop principal (rAF) ───────────────────────────────
      const loop = (now: DOMHighResTimeStamp) => {
        if (cancelledRef.current) return;
        if (video.ended) { finish(); return; }

        // Si está pausado (por auto-pausa o usuario), seguimos el rAF
        // pero no capturamos frames
        if (!video.paused && !inferring.current && now - lastInfTs.current >= minIntervalMs) {
          lastInfTs.current = now;
          inferring.current = true;

          // Captura del frame actual (Stretch / Squash)
          // Si el modelo fue entrenado en Roboflow sin "Fit (Padding)", aprendió a ver las imágenes estiradas
          try { 
            ctx.drawImage(video, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE); 
          }
          catch { inferring.current = false; rafRef.current = requestAnimationFrame(loop); return; }

          const ts = video.currentTime;

          onnxService.runInference(cvs, config.confidenceThreshold)
            .then(({ detections: raw, lowConfDetections, inferenceMs }) => {
              inferring.current = false;
              if (cancelledRef.current) return;

              const correctedRaw = raw;

              // ByteTrack de dos etapas: alta confianza para activos, baja para rescate
              const activeTracks = tracker.current.update(correctedRaw, lowConfDetections);
              
              // 📊 LOG DE DATOS DETALLADO PARA TABLAS
              if (activeTracks.length > 0) {
                const modelStr = onnxService.getActiveModelId() === 'ext' ? 'Extranjero' : 'ASGF';
                activeTracks.forEach(t => {
                  console.info(
                    `[TRACKER_DATA] Time: ${ts.toFixed(2)}s | Model: ${modelStr} | TrackID: ${t.id} | Class: ${t.className} | Conf: ${(t.confidence * 100).toFixed(1)}% | BBox: [${t.bbox.x1.toFixed(3)}, ${t.bbox.y1.toFixed(3)}, ${t.bbox.x2.toFixed(3)}, ${t.bbox.y2.toFixed(3)}]`
                  );
                });
              }

              setTracks([...activeTracks]);

              // Acumular detecciones confirmadas para export
              for (const t of activeTracks) {
                const det: Detection = {
                  bbox: t.bbox, classId: t.classId,
                  className: t.className, confidence: t.confidence,
                  timestamp: ts, frameIndex: 0,
                };
                allDetections.push(det);
                addDetection(det);
              }

              setProgress({
                currentFrame:     Math.round(ts * config.analysisFps),
                totalFrames:      Math.ceil(video.duration * config.analysisFps),
                currentTimestamp: ts,
                videoDuration:    video.duration,
                fps:              config.analysisFps,
                etaSeconds:       Math.round((video.duration - ts)),
                detectionsFound:  allDetections.length,
                lastInferenceMs:  inferenceMs,
              });

              // ── Actualización de TrackRegisters ──
              const currentStore = useAnalysisStore.getState();
              const currentTrackRegisters = currentStore.trackRegisters;
              const allTracksInTracker = tracker.current.getAllTracks();
              const activeIds = new Set<number>();
              
              for (const t of allTracksInTracker) {
                activeIds.add(t.id);
                const existingTrack = currentTrackRegisters[t.id];
                const isNew = !existingTrack;
                const needsThumbnail = !existingTrack || !existingTrack.thumbnailDataUrl;
                
                let thumbnailDataUrl = '';
                let thumbnailBbox = undefined;

                if (needsThumbnail) {
                  // Capturar el FRAME COMPLETO con la caja de detección dibujada
                  const tCvs = document.createElement('canvas');
                  const tCtx = tCvs.getContext('2d')!;
                  // Tamaño de miniatura más grande para ver contexto
                  tCvs.width  = 320;
                  tCvs.height = 320;

                  try {
                    // 1. Copiar el frame completo
                    tCtx.drawImage(cvs, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE, 0, 0, 320, 320);

                    // 2. Dibujar la caja de detección encima
                    const scale = 320 / CAPTURE_SIZE;
                    const bx1 = t.bbox.x1 * CAPTURE_SIZE * scale;
                    const by1 = t.bbox.y1 * CAPTURE_SIZE * scale;
                    const bx2 = t.bbox.x2 * CAPTURE_SIZE * scale;
                    const by2 = t.bbox.y2 * CAPTURE_SIZE * scale;
                    const bw  = bx2 - bx1;
                    const bh  = by2 - by1;

                    // Caja exterior con fill semitransparente
                    tCtx.strokeStyle = '#22c55e';
                    tCtx.lineWidth   = 2.5;
                    tCtx.strokeRect(bx1, by1, bw, bh);
                    tCtx.fillStyle   = 'rgba(34,197,94,0.15)';
                    tCtx.fillRect(bx1, by1, bw, bh);

                    // Esquinas estilo dron
                    const cs = Math.min(bw, bh) * 0.25;
                    tCtx.strokeStyle = '#ffffff';
                    tCtx.lineWidth   = 2;
                    tCtx.beginPath();
                    tCtx.moveTo(bx1, by1 + cs); tCtx.lineTo(bx1, by1); tCtx.lineTo(bx1 + cs, by1);
                    tCtx.moveTo(bx2 - cs, by1); tCtx.lineTo(bx2, by1); tCtx.lineTo(bx2, by1 + cs);
                    tCtx.moveTo(bx2, by2 - cs); tCtx.lineTo(bx2, by2); tCtx.lineTo(bx2 - cs, by2);
                    tCtx.moveTo(bx1 + cs, by2); tCtx.lineTo(bx1, by2); tCtx.lineTo(bx1, by2 - cs);
                    tCtx.stroke();

                    // Etiqueta con clase y confianza
                    const label = `${t.className} ${(t.confidence * 100).toFixed(0)}%`;
                    tCtx.font      = 'bold 11px monospace';
                    const tw = tCtx.measureText(label).width + 8;
                    const labelY = by1 > 18 ? by1 - 4 : by2 + 16;
                    tCtx.fillStyle = 'rgba(34,197,94,0.85)';
                    tCtx.beginPath();
                    tCtx.roundRect(bx1, labelY - 14, tw, 16, 3);
                    tCtx.fill();
                    tCtx.fillStyle = '#fff';
                    tCtx.fillText(label, bx1 + 4, labelY - 1);

                    thumbnailDataUrl = tCvs.toDataURL('image/jpeg', 0.85);
                    // thumbnailBbox ya no se usa (captura completa)
                    thumbnailBbox = undefined;
                  } catch (e) {
                    console.error('[TRACKER] Error generating thumbnail:', e);
                  }
                }

                upsertTrackRegister({
                  id: t.id,
                  className: t.className,
                  firstSeen: isNew ? ts : existingTrack.firstSeen,
                  lastSeen: ts,
                  maxConfidence: t.confidence,
                  status: 'active',
                  ...(thumbnailDataUrl ? { thumbnailDataUrl, thumbnailBbox } : {})
                });
              }

              markMissingTracksAsFinished(activeIds);
            })
            .catch((err) => { 
              console.error('[TRACKER] Inference error:', err);
              inferring.current = false; 
            });
        }

        rafRef.current = requestAnimationFrame(loop);
      };

      const finish = () => {
        cancelAnimationFrame(rafRef.current);
        if (cancelledRef.current) return;
        const totalMs = performance.now() - tStart;
        const avgConf = allDetections.length > 0
          ? allDetections.reduce((s, d) => s + d.confidence, 0) / allDetections.length
          : 0;
        setResult({
          detections:          allDetections,
          totalFramesAnalyzed: allDetections.length,
          videoDuration:       video.duration,
          processingTimeMs:    totalMs,
          averageConfidence:   avgConf,
          realFps:             config.analysisFps,
        });
        setTracks([]);
        setStatus('complete');
      };

      video.addEventListener('ended', finish, { once: true });
      rafRef.current = requestAnimationFrame(loop);

    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus('error');
    }
  }, [videoRef, config, ensureModel, resetAnalysis, setStatus, setProgress,
      addDetection, setResult, setError, setTracks,
      trackRegisters, upsertTrackRegister, markMissingTracksAsFinished]);

  // ── Cancelar ─────────────────────────────────────────────────
  const cancelAnalysis = useCallback(() => {
    cancelledRef.current = true;
    cancelAnimationFrame(rafRef.current);
    videoRef.current?.pause();
    tracker.current.reset();
    setTracks([]);
    setStatus('cancelled');
  }, [setStatus, setTracks]);

  return { startAnalysis, cancelAnalysis };
}
