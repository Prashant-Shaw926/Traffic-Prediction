import type { CongestionLevel, Junction, PredictionPoint } from '../types/traffic'
import { iterateWindow, toDateTimeLocal } from '../utils/horizon'

export const JUNCTIONS: Junction[] = [
  {
    id: 'j1',
    name: 'Junction 1',
    area: 'Silk Board',
    lat: 12.9177,
    lng: 77.6238,
  },
  {
    id: 'j2',
    name: 'Junction 2',
    area: 'HSR 27th Main',
    lat: 12.9121,
    lng: 77.6446,
  },
  {
    id: 'j3',
    name: 'Junction 3',
    area: 'Koramangala 80 ft',
    lat: 12.9352,
    lng: 77.6245,
  },
  {
    id: 'j4',
    name: 'Junction 4',
    area: 'Agara',
    lat: 12.9246,
    lng: 77.6489,
  },
]

const BASE_VOLUME: Record<string, number> = {
  j1: 110,
  j2: 82,
  j3: 64,
  j4: 48,
}

function peak(hour: number, center: number, width: number, height: number): number {
  const z = (hour - center) / width
  return height * Math.exp(-0.5 * z * z)
}

function hourFactor(hour: number, isWeekend: boolean): number {
  const night = 0.18
  if (isWeekend) {
    return night + peak(hour, 13, 3.2, 0.5) + peak(hour, 18, 2.5, 0.28)
  }
  return night + peak(hour, 9, 1.1, 0.85) + peak(hour, 18.5, 1.4, 0.92)
}

function hash01(seed: string): number {
  let hash = 2166136261
  for (let i = 0; i < seed.length; i += 1) {
    hash ^= seed.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0) / 4294967296
}

export function congestionFromVolume(volume: number): CongestionLevel {
  if (volume < 26) return 'clear'
  if (volume < 51) return 'moderate'
  if (volume < 81) return 'heavy'
  return 'severe'
}

export function volumeAt(date: Date, junctionId: string): number {
  const base = BASE_VOLUME[junctionId] ?? 60
  const hour = date.getHours() + date.getMinutes() / 60
  const isWeekend = date.getDay() === 0 || date.getDay() === 6
  return Math.max(0, Math.round(base * hourFactor(hour, isWeekend)))
}

export function actualFromPredicted(predicted: number, seed: string): number {
  const noise = hash01(seed)
  return Math.max(0, Math.round(predicted * (0.88 + noise * 0.24)))
}

export function buildSeries(
  locationId: string,
  start: string,
  end: string,
  includeActual: boolean,
): PredictionPoint[] {
  return iterateWindow(start, end).map((date) => {
    const timestamp = toDateTimeLocal(date)
    const predicted = volumeAt(date, locationId)
    const actual = includeActual
      ? actualFromPredicted(predicted, `${locationId}:${timestamp}`)
      : undefined
    const congestion = congestionFromVolume(actual ?? predicted)
    return { timestamp, predicted, actual, congestion }
  })
}

export function peakOf(points: PredictionPoint[]): {
  peakVolume: number
  peakCongestion: CongestionLevel
} {
  const peakVolume = points.reduce(
    (max, point) => Math.max(max, point.predicted),
    0,
  )
  return {
    peakVolume,
    peakCongestion: congestionFromVolume(peakVolume),
  }
}
