import { useRef, useCallback, RefObject } from 'react';
import * as onnxService from '../services/onnxService';
import type { Detection } from '../types';
import { useAnalysisStore } from '../store/analysisStore';
import type { ModelId } from '../store/analysisStore';

const CAPTURE_SIZE = 640;

export function useImageAnalysis(
  imageRef: RefObject<HTMLImageElement | null>
) {
  const cancelledRef = useRef(false);
  const canvas = useRef<HTMLCanvasElement | null>(null);

  const {
    config, setStatus, setProgress,
    addDetection, setResult, setError, setHardware,
    resetAnalysis, setTracks,
  } = useAnalysisStore();

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

  const startAnalysis = useCallback(async () => {
    const img = imageRef.current;
    if (!img?.src) {
      setError('Selecciona una imagen antes de analizar.');
      setStatus('error');
      return;
    }

    resetAnalysis();
    cancelledRef.current = false;

    try {
      const currentModelId = useAnalysisStore.getState().activeModelId;
      await ensureModel(currentModelId);
      if (cancelledRef.current) { setStatus('cancelled'); return; }

      if (!canvas.current) {
        canvas.current = document.createElement('canvas');
        canvas.current.width = canvas.current.height = CAPTURE_SIZE;
      }
      const cvs = canvas.current;
      const ctx = cvs.getContext('2d', { willReadFrequently: true })!;

      const tStart = performance.now();

      if (!img.complete || img.naturalWidth === 0) {
        await new Promise((resolve) => {
          img.onload = resolve;
        });
      }

      ctx.drawImage(img, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE);

      setStatus('analyzing');

      const { detections, inferenceMs } = await onnxService.runInference(cvs, config.confidenceThreshold);
      if (cancelledRef.current) return;

      const avgConf = detections.length > 0
        ? detections.reduce((s, d) => s + d.confidence, 0) / detections.length
        : 0;

      const tracks = detections.map((d, i) => ({
        id: i + 1,
        bbox: d.bbox,
        classId: d.classId,
        className: d.className,
        confidence: d.confidence,
        age: 1,
        hits: 1,
        trail: [] as [number, number][],
      }));

      setTracks(tracks);
      
      for (const d of detections) addDetection(d);

      const totalMs = performance.now() - tStart;
      
      setProgress({
        currentFrame: 1,
        totalFrames: 1,
        currentTimestamp: 0,
        videoDuration: 0,
        fps: 1,
        etaSeconds: 0,
        detectionsFound: detections.length,
        lastInferenceMs: inferenceMs,
      });

      setResult({
        detections,
        totalFramesAnalyzed: 1,
        videoDuration: 0,
        processingTimeMs: totalMs,
        averageConfidence: avgConf,
        realFps: 1,
      });

      setStatus('complete');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatus('error');
    }
  }, [imageRef, config, ensureModel, resetAnalysis, setStatus, setProgress, addDetection, setResult, setError, setTracks]);

  const cancelAnalysis = useCallback(() => {
    cancelledRef.current = true;
    setTracks([]);
    setStatus('cancelled');
  }, [setStatus, setTracks]);

  return { startAnalysis, cancelAnalysis };
}
