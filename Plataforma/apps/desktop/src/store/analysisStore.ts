import { create } from 'zustand';
import type {
  AnalysisConfig,
  AnalysisResult,
  AnalysisStatus,
  Detection,
  HardwareInfo,
  ProgressUpdate,
  Track,
  TrackRegister,
} from '../types';

export type ModelId = 'ext' | 'asgf' | 'v2' | 'v3';

interface AnalysisStore {
  mediaType: 'video' | 'image' | null;
  videoPath: string | null;
  videoName: string | null;
  imagePath: string | null;
  imageName: string | null;
  videoDuration: number;

  activeModelId: ModelId;    // modelo activo seleccionado por el usuario

  config: AnalysisConfig;
  status: AnalysisStatus;
  progress: ProgressUpdate | null;
  error: string | null;
  hardware: HardwareInfo | null;

  detections: Detection[];   // historial completo para export
  tracks: Track[];       // tracks activos en el frame actual (overlay)
  result: AnalysisResult | null;

  trackRegisters: Record<number, TrackRegister>; // Historial y estado de rastros
  selectedTrackId: number | null; // ID del rastro seleccionado para resaltar en UI

  setVideoPath: (path: string, name: string) => void;
  setImagePath: (path: string, name: string) => void;
  setVideoDuration: (duration: number) => void;
  setConfig: (patch: Partial<AnalysisConfig>) => void;
  setStatus: (status: AnalysisStatus) => void;
  setProgress: (progress: ProgressUpdate) => void;
  addDetection: (detection: Detection) => void;
  setTracks: (tracks: Track[]) => void;
  setResult: (result: AnalysisResult) => void;
  setError: (error: string | null) => void;
  setHardware: (info: HardwareInfo) => void;
  setActiveModelId: (id: ModelId) => void;
  resetAnalysis: () => void;
  clearVideo: () => void;

  upsertTrackRegister: (register: Partial<TrackRegister> & { id: number }) => void;
  markMissingTracksAsFinished: (activeIds: Set<number>) => void;
  setSelectedTrackId: (id: number | null) => void;
}

const DEFAULT_CONFIG: AnalysisConfig = {
  analysisFps: 5,
  confidenceThreshold: 0.25,
};

export const useAnalysisStore = create<AnalysisStore>((set) => ({
  mediaType: null,
  videoPath: null,
  videoName: null,
  imagePath: null,
  imageName: null,
  videoDuration: 0,

  activeModelId: 'ext',   // empieza con el modelo extranjero por compatibilidad

  config: DEFAULT_CONFIG,
  status: 'idle',
  progress: null,
  error: null,
  hardware: null,

  detections: [],
  tracks: [],
  result: null,
  trackRegisters: {},
  selectedTrackId: null,

  setVideoPath: (path, name) => set({ videoPath: path, videoName: name, mediaType: 'video', imagePath: null, imageName: null }),
  setImagePath: (path, name) => set({ imagePath: path, imageName: name, mediaType: 'image', videoPath: null, videoName: null }),
  setVideoDuration: (duration) => set({ videoDuration: duration }),
  setConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),
  setStatus: (status) => set({ status }),
  setProgress: (progress) => set({ progress }),
  addDetection: (d) => set((s) => ({ detections: [...s.detections, d] })),
  setTracks: (tracks) => set({ tracks }),
  setResult: (result) => set({ result }),
  setError: (error) => set({ error }),
  setHardware: (hardware) => set({ hardware }),
  setActiveModelId: (id) => set({ activeModelId: id }),

  resetAnalysis: () => set({
    status: 'idle',
    progress: null,
    error: null,
    detections: [],
    tracks: [],
    result: null,
    trackRegisters: {},
    selectedTrackId: null,
  }),

  clearVideo: () => set({
    mediaType: null,
    videoPath: null,
    videoName: null,
    imagePath: null,
    imageName: null,
    videoDuration: 0,
    status: 'idle',
    progress: null,
    error: null,
    detections: [],
    tracks: [],
    result: null,
    trackRegisters: {},
    selectedTrackId: null,
  }),

  upsertTrackRegister: (patch) => set((s) => {
    const existing = s.trackRegisters[patch.id];
    if (existing) {
      return {
        trackRegisters: {
          ...s.trackRegisters,
          [patch.id]: {
            ...existing,
            ...patch,
            maxConfidence: Math.max(existing.maxConfidence, patch.maxConfidence ?? 0),
          },
        },
      };
    }
    return {
      trackRegisters: {
        ...s.trackRegisters,
        [patch.id]: patch as TrackRegister,
      },
    };
  }),

  markMissingTracksAsFinished: (activeIds) => set((s) => {
    let changed = false;
    const next = { ...s.trackRegisters };
    for (const key of Object.keys(next)) {
      const id = Number(key);
      if (next[id].status === 'active' && !activeIds.has(id)) {
        next[id] = { ...next[id], status: 'finished' };
        changed = true;
      }
    }
    return changed ? { trackRegisters: next } : {};
  }),

  setSelectedTrackId: (id) => set({ selectedTrackId: id }),
}));
