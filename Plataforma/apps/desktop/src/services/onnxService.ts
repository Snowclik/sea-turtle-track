import * as ort from 'onnxruntime-web/wasm'
import type { Detection } from '../types'
import type { ModelId } from '../store/analysisStore'

// WASM files servidos desde la raíz del frontend
ort.env.wasm.wasmPaths = '/'
ort.env.wasm.numThreads = 1

// Rutas dinámicas por modelo
const MODEL_CONFIGS: Record<ModelId, { onnx: string; meta: string }> = {
  ext:  { onnx: '/models/model_ext.onnx',  meta: '/models/model_ext_meta.json'  },
  asgf: { onnx: '/models/model_asgf.onnx', meta: '/models/model_asgf_meta.json' },
  v2:   { onnx: '/models/model_v2.onnx',   meta: '/models/model_v2_meta.json'   },
  v3:   { onnx: '/models/model_v3.onnx',   meta: '/models/model_v3_meta.json'   },
}

const INPUT_SIZE = 640

let session:       ort.InferenceSession | null = null
let classNames:    string[]                    = ['object']
let activeModelId: ModelId | null             = null

// ─────────────────────────────────────────────────────────────
// Carga del modelo — destruye sesión anterior si existe
// ─────────────────────────────────────────────────────────────

export async function releaseSession(): Promise<void> {
  if (session) {
    try { await session.release() } catch { /* noop */ }
    session       = null
    activeModelId = null
  }
}

export async function loadModel(
  modelId: ModelId,
): Promise<{ classNames: string[]; numClasses: number }> {
  // Si ya está cargado el mismo modelo, no hacer nada
  if (session && activeModelId === modelId) {
    return { classNames, numClasses: classNames.length }
  }

  // Destruir sesión anterior para liberar memoria WASM
  await releaseSession()

  const cfg = MODEL_CONFIGS[modelId]

  // Cargar metadatos dinámicamente desde JSON
  classNames = ['object']
  try {
    const res  = await fetch(cfg.meta)
    const meta = await res.json()
    if (Array.isArray(meta.class_names)) classNames = meta.class_names
  } catch {
    console.warn(`[OnnxService] ${cfg.meta} no encontrado, usando clase por defecto`)
  }

  session = await ort.InferenceSession.create(cfg.onnx, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })

  activeModelId = modelId
  return { classNames, numClasses: classNames.length }
}

export function isModelLoaded(modelId: ModelId): boolean {
  return session !== null && activeModelId === modelId
}

export function getActiveModelId(): ModelId | null {
  return activeModelId
}

// ─────────────────────────────────────────────────────────────
// Inferencia — stride dinámico: soporta bbox [1,300,6]
//              y seg [1,300,38] con el mismo código
// ─────────────────────────────────────────────────────────────

// Umbral bajo fijo para ByteTrack — el tracker lo usa internamente para rescatar tracks perdidos
const LOW_CONF_THRESH = 0.10

export async function runInference(
  canvas: HTMLCanvasElement,
  confThreshold: number,  // umbral alto (lo que muestra el usuario)
): Promise<{ detections: Detection[]; lowConfDetections: Detection[]; inferenceMs: number }> {
  if (!session) throw new Error('Modelo no cargado')

  const tensor = canvasToTensor(canvas)
  const input  = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE])

  const t0      = performance.now()
  const outputs = await session.run({ images: input })
  const inferenceMs = performance.now() - t0

  // output0: [1, maxDets, stride]
  //   stride = 6  → bbox model (ext)
  //   stride = 38 → seg model (asgf) — usamos solo los primeros 6 valores
  const out    = outputs['output0']
  const data   = out.data as Float32Array
  const maxDets = out.dims[1] as number
  const stride  = out.dims[2] as number   // ← stride dinámico
  const inv     = 1.0 / INPUT_SIZE

  const detections: Detection[]     = []  // alta confianza
  const lowConfDetections: Detection[] = [] // baja confianza (para ByteTrack stage 2)

  for (let i = 0; i < maxDets; i++) {
    const base = i * stride
    const conf = data[base + 4]
    // Ignorar absolutamente todo lo que esté por debajo del umbral mínimo
    if (conf < LOW_CONF_THRESH) continue

    const x1 = data[base + 0]
    const y1 = data[base + 1]
    const x2 = data[base + 2]
    const y2 = data[base + 3]
    if (x2 <= x1 || y2 <= y1) continue

    const classId   = Math.round(data[base + 5])

    const className = classNames[classId] ?? `clase_${classId}`

    // En v3 filtrar Hojas (ruido)
    if (activeModelId === 'v3' && className === 'Hojas') continue

    const det: Detection = {
      bbox: {
        x1: Math.max(0, Math.min(1, x1 * inv)),
        y1: Math.max(0, Math.min(1, y1 * inv)),
        x2: Math.max(0, Math.min(1, x2 * inv)),
        y2: Math.max(0, Math.min(1, y2 * inv)),
      },
      classId,
      className,
      confidence: conf,
      timestamp:  0,
      frameIndex: 0,
    }

    if (conf >= confThreshold) {
      detections.push(det)           // alta confianza → se muestra
    } else {
      lowConfDetections.push(det)    // baja confianza → solo para rescatar tracks
    }
  }

  return { detections, lowConfDetections, inferenceMs }
}

// ─────────────────────────────────────────────────────────────
// Preprocesamiento: canvas 640×640 → tensor Float32 [1,3,640,640]
// ─────────────────────────────────────────────────────────────

function canvasToTensor(canvas: HTMLCanvasElement): Float32Array {
  const ctx    = canvas.getContext('2d', { willReadFrequently: true })!
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)
  const pixels = INPUT_SIZE * INPUT_SIZE
  const tensor = new Float32Array(3 * pixels)

  for (let i = 0; i < pixels; i++) {
    tensor[i]              = data[i * 4]     / 255  // R
    tensor[pixels + i]     = data[i * 4 + 1] / 255  // G
    tensor[2 * pixels + i] = data[i * 4 + 2] / 255  // B
  }

  return tensor
}
