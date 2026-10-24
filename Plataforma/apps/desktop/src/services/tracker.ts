import type { BoundingBox } from '../types'

// ─────────────────────────────────────────────────────────────
// ByteTrack-lite en TypeScript puro
// Referencia: github.com/FoundationVision/ByteTrack
//
// La idea clave vs SORT:
//   SORT: descarta detecciones de baja confianza → pierde tracks
//   ByteTrack: usa detecciones de baja confianza en una SEGUNDA
//              etapa SOLO para rescatar tracks que estaban perdidos
// ─────────────────────────────────────────────────────────────

const MAX_LOST  = 90   // frames en estado "perdido" antes de eliminar (≈3s a 30fps)
const MIN_HITS  = 1    // detecciones mínimas para confirmar y mostrar un track
const HIGH_DIST = 0.35 // dist. centroide para asociar detección alta conf con track activo
const LOW_DIST  = 0.55 // dist. más amplia para rescatar track perdido con detección baja conf
const NMS_THRESH = 0.20 // IoU mínimo para suprimir detección duplicada
const NMS_CDIST  = 0.07 // o centroides a menos de 7% de pantalla

let _nextId = 1

// ─────────────────────────────────────────────────────────────
// Tipos
// ─────────────────────────────────────────────────────────────

type TrackState = 'tracked' | 'lost'

export interface Track {
  id:         number
  bbox:       BoundingBox
  classId:    number
  className:  string
  confidence: number
  age:        number           // frames desde el último match
  hits:       number
  state:      TrackState
  trail:      [number, number][]
}

export interface RawDetection {
  bbox:       BoundingBox
  classId:    number
  className:  string
  confidence: number
}

// ─────────────────────────────────────────────────────────────
// Utilidades
// ─────────────────────────────────────────────────────────────

function centDist(a: BoundingBox, b: BoundingBox): number {
  const ax = (a.x1 + a.x2) / 2, ay = (a.y1 + a.y2) / 2
  const bx = (b.x1 + b.x2) / 2, by = (b.y1 + b.y2) / 2
  return Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
}

function iou(a: BoundingBox, b: BoundingBox): number {
  const ix1 = Math.max(a.x1, b.x1), iy1 = Math.max(a.y1, b.y1)
  const ix2 = Math.min(a.x2, b.x2), iy2 = Math.min(a.y2, b.y2)
  if (ix2 <= ix1 || iy2 <= iy1) return 0
  const inter = (ix2 - ix1) * (iy2 - iy1)
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter)
}

/** NMS sobre detecciones del mismo frame, usando IoU Y distancia de centroide */
function nms(dets: RawDetection[]): RawDetection[] {
  const sorted = [...dets].sort((a, b) => b.confidence - a.confidence)
  const keep   = new Array(sorted.length).fill(true)
  for (let i = 0; i < sorted.length; i++) {
    if (!keep[i]) continue
    for (let j = i + 1; j < sorted.length; j++) {
      if (!keep[j]) continue
      if (sorted[i].classId !== sorted[j].classId) continue
      if (iou(sorted[i].bbox, sorted[j].bbox) > NMS_THRESH ||
          centDist(sorted[i].bbox, sorted[j].bbox) < NMS_CDIST) {
        keep[j] = false
      }
    }
  }
  return sorted.filter((_, i) => keep[i])
}

/** Algoritmo húngaro greedy (O(n²)) — suficiente para ≤20 tracks */
function greedyMatch(
  tracks: Track[],
  dets:   RawDetection[],
  maxDist: number,
): { ti: number; di: number }[] {
  // Construir matriz de costes (distancia centroide; ∞ si clase diferente)
  const costs: number[][] = tracks.map(trk =>
    dets.map(det =>
      trk.classId !== det.classId ? Infinity : centDist(trk.bbox, det.bbox)
    )
  )

  const usedDet = new Set<number>()
  const usedTrk = new Set<number>()
  const pairs: { ti: number; di: number }[] = []

  // Greedy: escoger el par de menor coste primero
  while (true) {
    let best = maxDist
    let bti = -1, bdi = -1
    for (let ti = 0; ti < tracks.length; ti++) {
      if (usedTrk.has(ti)) continue
      for (let di = 0; di < dets.length; di++) {
        if (usedDet.has(di)) continue
        if (costs[ti][di] < best) { best = costs[ti][di]; bti = ti; bdi = di }
      }
    }
    if (bti === -1) break
    pairs.push({ ti: bti, di: bdi })
    usedTrk.add(bti); usedDet.add(bdi)
  }

  return pairs
}

// ─────────────────────────────────────────────────────────────
// ByteTracker
// ─────────────────────────────────────────────────────────────

export class IoUTracker {
  private tracks: Track[] = []

  /**
   * @param highDets  Detecciones de alta confianza (umbral del usuario)
   * @param lowDets   Detecciones de baja confianza (solo para rescate interno)
   */
  update(highDets: RawDetection[], lowDets: RawDetection[] = []): Track[] {
    // NMS sobre alta y baja confianza por separado
    const hiDets = nms(highDets)
    const loDets = nms(lowDets)

    const activeTracks = this.tracks.filter(t => t.state === 'tracked')
    const lostTracks   = this.tracks.filter(t => t.state === 'lost')

    // ── ETAPA 1: Alta confianza → tracks ACTIVOS ─────────────
    const pairs1   = greedyMatch(activeTracks, hiDets, HIGH_DIST)
    const matchedTi1 = new Set(pairs1.map(p => p.ti))
    const matchedDi1 = new Set(pairs1.map(p => p.di))

    // Actualizar tracks emparejados
    for (const { ti, di } of pairs1) {
      const trk = activeTracks[ti]
      const det = hiDets[di]
      const cx  = (det.bbox.x1 + det.bbox.x2) / 2
      const cy  = (det.bbox.y1 + det.bbox.y2) / 2
      const idx = this.tracks.indexOf(trk)
      this.tracks[idx] = {
        ...trk,
        bbox:       det.bbox,
        confidence: det.confidence,
        age:        0,
        hits:       trk.hits + 1,
        state:      'tracked',
        trail:      [...trk.trail.slice(-20), [cx, cy]],
      }
    }

    // Tracks activos sin match → pasan a estado "perdido"
    for (let ti = 0; ti < activeTracks.length; ti++) {
      if (matchedTi1.has(ti)) continue
      const idx = this.tracks.indexOf(activeTracks[ti])
      this.tracks[idx].state = 'lost'
      this.tracks[idx].age++
    }

    // Detecciones altas sin match → pueden rescatar tracks perdidos (etapa 2a)
    const unmatchedHi = hiDets.filter((_, i) => !matchedDi1.has(i))

    // ── ETAPA 2: Detecciones sin match → tracks PERDIDOS ─────
    // Combinamos unmatchedHi + loDets para el rescate
    const rescueDets = [...unmatchedHi, ...loDets]
    const pairs2     = greedyMatch(lostTracks, rescueDets, LOW_DIST)
    const matchedTi2 = new Set(pairs2.map(p => p.ti))
    const matchedDi2 = new Set(pairs2.map(p => p.di))

    for (const { ti, di } of pairs2) {
      const trk = lostTracks[ti]
      const det = rescueDets[di]
      const cx  = (det.bbox.x1 + det.bbox.x2) / 2
      const cy  = (det.bbox.y1 + det.bbox.y2) / 2
      const idx = this.tracks.indexOf(trk)
      // ¡Resucitar con el mismo ID!
      this.tracks[idx] = {
        ...trk,
        bbox:       det.bbox,
        confidence: det.confidence,
        age:        0,
        hits:       trk.hits + 1,
        state:      'tracked',   // vuelve a activo
        trail:      [...trk.trail.slice(-20), [cx, cy]],
      }
    }

    // Tracks perdidos que siguen sin match → envejecer
    for (let ti = 0; ti < lostTracks.length; ti++) {
      if (matchedTi2.has(ti)) continue
      const idx = this.tracks.indexOf(lostTracks[ti])
      this.tracks[idx].age++
    }

    // ── ETAPA 3: Detecciones nuevas → tracks nuevos ──────────
    // Detecciones de alta confianza que no emparejaron con nada
    const unmatchedForNew = rescueDets.filter((_, i) => !matchedDi2.has(i))
    // Solo las que eran de alta confianza generan nuevos tracks
    for (const det of unmatchedForNew) {
      // Solo crear track nuevo si la detección es de alta confianza
      if (!hiDets.includes(det) && !unmatchedHi.includes(det)) continue
      const cx = (det.bbox.x1 + det.bbox.x2) / 2
      const cy = (det.bbox.y1 + det.bbox.y2) / 2
      this.tracks.push({
        id:         _nextId++,
        bbox:       det.bbox,
        classId:    det.classId,
        className:  det.className,
        confidence: det.confidence,
        age:        0,
        hits:       1,
        state:      'tracked',
        trail:      [[cx, cy]],
      })
    }

    // ── LIMPIEZA: eliminar tracks muy viejos ──────────────────
    this.tracks = this.tracks.filter(t => t.age <= MAX_LOST)

    // Retornar solo tracks activos, confirmados y vistos en este frame
    return this.tracks.filter(t => t.state === 'tracked' && t.hits >= MIN_HITS && t.age === 0)
  }

  reset() {
    this.tracks = []
    _nextId = 1
  }

  getAllTracks(): Track[] {
    return this.tracks
  }
}
