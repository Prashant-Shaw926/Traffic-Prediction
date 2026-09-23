export type CongestionLevel = 'clear' | 'moderate' | 'heavy' | 'severe'

export type Junction = {
  id: string
  name: string
  area: string
  lat: number
  lng: number
}

export type PredictionPoint = {
  timestamp: string
  predicted: number
  actual?: number
  congestion: CongestionLevel
}

export type PredictRequest = {
  locationId: string
  start: string
  end: string
}

export type PredictResponse = {
  locationId: string
  start: string
  end: string
  points: PredictionPoint[]
  peakCongestion: CongestionLevel
  peakVolume: number
}

export type TrafficApi = {
  getLocations: () => Promise<Junction[]>
  predict: (request: PredictRequest) => Promise<PredictResponse>
  getHistory: (
    locationId: string,
    start: string,
    end: string,
  ) => Promise<PredictionPoint[]>
}

export const CONGESTION_COLORS: Record<CongestionLevel, string> = {
  clear: '#3dd68c',
  moderate: '#e8c547',
  heavy: '#f76707',
  severe: '#e03131',
}

export const CONGESTION_LABELS: Record<CongestionLevel, string> = {
  clear: 'Clear',
  moderate: 'Moderate',
  heavy: 'Heavy',
  severe: 'Severe',
}

export const CONGESTION_ORDER: CongestionLevel[] = [
  'clear',
  'moderate',
  'heavy',
  'severe',
]
