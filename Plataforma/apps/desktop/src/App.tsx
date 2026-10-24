import { useEffect, useRef, useState } from 'react';
import { VideoPlayer } from './components/VideoPlayer';
import { ImageViewer } from './components/ImageViewer';
import { ControlBar }  from './components/ControlBar';
import { StatsPanel }  from './components/StatsPanel';
import { TracksPanel } from './components/TracksPanel';
import { useAnalysisStore } from './store/analysisStore';
import type { ModelId } from './store/analysisStore';
import * as onnxService from './services/onnxService';
import { Badge } from '@/components/ui/badge';
import { Cpu, ScanLine, Video, Scan, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/utils';

const MODEL_OPTIONS: { id: ModelId; label: string; sub: string }[] = [
  { id: 'ext',  label: 'Modelo Base (Ext)',             sub: 'Filtra todo menos Track' },
  { id: 'asgf', label: 'ASGF Nidos',                    sub: 'Segmentación de nidos (Roboflow)' },
  { id: 'v2',   label: 'Tortugas AI v2',                sub: 'Dataset Roboflow (205gp 3seqi)' },
  { id: 'v3',   label: 'Tortugas AI v3',                sub: 'Dataset Roboflow (Ángulos sintéticos)' },
];

type MobileView = 'visor' | 'rastros' | 'metricas';

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);

  const [mobileView, setMobileView] = useState<MobileView>('visor');

  const { activeModelId, setActiveModelId, resetAnalysis, setHardware, mediaType, trackRegisters } = useAnalysisStore();

  const tracksCount = Object.keys(trackRegisters).length;

  // Inicialización: cargar HW info con el modelo por defecto
  useEffect(() => {
    console.info(`[TRACKER] [${new Date().toLocaleTimeString()}] 🚀 Plataforma iniciada. Modelo por defecto: ${activeModelId}`);
    setHardware({
      executionProvider: 'WASM (onnxruntime-web)',
      modelLoaded: false,
      classNames: ['Fox', 'Track', 'Turtle'],
      numClasses: 3,
    });
  }, []);

  // Cambio de modelo: liberar sesión anterior y limpiar estado
  const handleModelChange = async (newId: ModelId) => {
    if (newId === activeModelId) return;

    const modelName = MODEL_OPTIONS.find(m => m.id === newId)?.label || newId;
    console.info(`[TRACKER] [${new Date().toLocaleTimeString()}] 🔄 Cambiando modelo IA activo a: ${modelName}`);

    // Pausar video si está reproduciendo
    videoRef.current?.pause();

    // Liberar sesión ONNX anterior
    try { await (onnxService as any).releaseSession?.(); } catch { /* noop */ }

    // Actualizar store
    setActiveModelId(newId);
    resetAnalysis();

    // Actualizar hw info placeholder hasta que se cargue el modelo real
    setHardware({
      executionProvider: 'WASM (onnxruntime-web)',
      modelLoaded: false,
      classNames: newId === 'asgf' ? ['Nido'] : newId === 'v2' ? ['Fox', 'Hojas', 'Track', 'Turtle'] : newId === 'v3' ? ['Fox', 'Track', 'Turtle'] : ['Fox', 'Track', 'Turtle'],
      numClasses: newId === 'asgf' ? 1 : newId === 'v2' ? 4 : 3,
    });
  };

  return (
    <div className="min-h-screen h-dvh bg-[#fcfcfd] text-foreground flex flex-col font-sans selection:bg-primary selection:text-primary-foreground overflow-hidden">

      {/* Top Header — Autoajustable para Galaxy Fold / Móvil / Desktop */}
      <header className="h-14 px-3 sm:px-5 bg-background border-b border-border flex items-center justify-between shrink-0 sticky top-0 z-20">
        
        {/* Brand / Logo */}
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          <div className="h-8 w-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center shadow-xs shrink-0">
            <ScanLine className="w-4 h-4" />
          </div>
          <div className="flex flex-col min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="text-xs font-semibold tracking-tight text-foreground truncate">
                Tortugas AI
              </span>
              <span className="text-[10px] font-mono font-medium px-1 py-0.2 text-muted-foreground bg-secondary rounded border border-border/60 shrink-0">
                v2.0
              </span>
            </div>
            <span className="text-[10px] text-muted-foreground leading-none truncate hidden xs:block">
              Inspección UAV & Rastros
            </span>
          </div>
        </div>

        {/* Selector de Modelos: Dropdown en móvil/Fold cover screen, Pills en desktop */}
        <div className="flex items-center gap-2">
          {/* Dropdown compacto para móviles / Galaxy Fold cerrado */}
          <div className="block sm:hidden">
            <select
              value={activeModelId}
              onChange={(e) => handleModelChange(e.target.value as ModelId)}
              className="h-7 text-xs bg-secondary/80 border border-border rounded-md px-2 text-foreground font-medium outline-none focus:ring-1 focus:ring-ring cursor-pointer max-w-[130px] truncate"
            >
              {MODEL_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          {/* Segmented control para pantallas medianas y de escritorio */}
          <div className="hidden sm:flex items-center gap-2">
            <span className="text-[11px] text-muted-foreground font-medium hidden md:inline-block">
              Modelo:
            </span>
            <div className="p-0.5 bg-secondary/80 rounded-lg border border-border flex items-center gap-0.5">
              {MODEL_OPTIONS.map((opt) => {
                const isActive = activeModelId === opt.id;
                return (
                  <button
                    key={opt.id}
                    onClick={() => handleModelChange(opt.id)}
                    title={opt.sub}
                    className={`
                      px-2.5 py-1 rounded-md text-xs transition-all duration-150 cursor-pointer whitespace-nowrap
                      ${isActive
                        ? 'bg-background text-foreground font-medium shadow-xs border border-border/40'
                        : 'text-muted-foreground hover:text-foreground font-normal hover:bg-background/40'}
                    `}
                  >
                    {opt.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Hardware Status indicator */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-secondary/60 border border-border text-[11px] text-muted-foreground">
            <Cpu className="w-3.5 h-3.5 text-foreground/70" />
            <span className="font-mono text-[10px] hidden xs:inline">WASM</span>
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          </div>
        </div>
      </header>

      {/* Main Responsive Area */}
      <main className="flex-1 flex overflow-hidden relative">

        {/* Panel Izquierdo: Rastros Detectados */}
        {/* En móvil (< lg) se muestra si mobileView === 'rastros'; en Desktop (lg+) siempre visible */}
        <div
          className={cn(
            "h-full overflow-hidden shrink-0",
            mobileView === 'rastros' ? "flex flex-col w-full absolute inset-0 z-20 bg-background lg:static lg:w-72" : "hidden lg:flex lg:w-72"
          )}
        >
          <TracksPanel />
        </div>

        {/* Contenido Central: Visor de Video / Imagen + Barra de Control */}
        {/* En móvil (< lg) se muestra si mobileView === 'visor'; en Desktop (lg+) siempre visible */}
        <div
          className={cn(
            "flex-1 flex flex-col min-w-0 overflow-y-auto p-3 sm:p-4 gap-3 sm:gap-4",
            mobileView === 'visor' ? "flex w-full pb-24 lg:pb-4" : "hidden lg:flex"
          )}
        >
          <div className="flex-1 flex flex-col min-h-0 justify-center">
            {mediaType === 'image' ? (
              <ImageViewer imageRef={imageRef} />
            ) : (
              <VideoPlayer videoRef={videoRef} />
            )}
          </div>
          <ControlBar videoRef={videoRef} imageRef={imageRef} />
        </div>

        {/* Panel Derecho: Métricas y Exportación */}
        {/* En móvil (< lg) se muestra si mobileView === 'metricas'; en Desktop (lg+) siempre visible */}
        <aside
          className={cn(
            "overflow-y-auto p-3 sm:p-4 shrink-0 bg-background",
            mobileView === 'metricas'
              ? "block w-full absolute inset-0 z-20 pb-24 lg:pb-4 lg:static lg:w-80 lg:border-l lg:border-border"
              : "hidden lg:block lg:w-80 lg:border-l lg:border-border"
          )}
        >
          <StatsPanel />
        </aside>
      </main>

      {/* Dock Flotante estilo Apple Liquid Glass para Dispositivos Móviles y Plegables (< lg) */}
      <nav
        aria-label="Navegación móvil adaptable"
        className="fixed bottom-[max(0.85rem,env(safe-area-inset-bottom))] inset-x-0 z-40 mx-auto flex w-fit max-w-[calc(100vw-1.5rem)] items-center gap-1 rounded-full liquid-glass-dock p-1.5 lg:hidden shadow-lg select-none"
      >
        <button
          type="button"
          onClick={() => setMobileView('visor')}
          className={cn(
            "flex h-9 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-all duration-150 active:scale-95 cursor-pointer",
            mobileView === 'visor'
              ? "liquid-glass-active font-semibold text-foreground shadow-xs"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          <Video className="w-3.5 h-3.5" />
          <span>Visor</span>
        </button>

        <button
          type="button"
          onClick={() => setMobileView('rastros')}
          className={cn(
            "flex h-9 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-all duration-150 active:scale-95 cursor-pointer",
            mobileView === 'rastros'
              ? "liquid-glass-active font-semibold text-foreground shadow-xs"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          <Scan className="w-3.5 h-3.5" />
          <span>Rastros</span>
          {tracksCount > 0 && (
            <span className="text-[10px] font-mono font-bold px-1.5 py-0.2 rounded-full bg-primary text-primary-foreground">
              {tracksCount}
            </span>
          )}
        </button>

        <button
          type="button"
          onClick={() => setMobileView('metricas')}
          className={cn(
            "flex h-9 items-center gap-1.5 rounded-full px-3 text-xs font-medium transition-all duration-150 active:scale-95 cursor-pointer",
            mobileView === 'metricas'
              ? "liquid-glass-active font-semibold text-foreground shadow-xs"
              : "text-muted-foreground hover:text-foreground"
          )}
        >
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Métricas</span>
        </button>
      </nav>
    </div>
  );
}
