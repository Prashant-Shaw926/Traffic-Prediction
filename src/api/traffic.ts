import type {
  CongestionLevel,
  Junction,
  PredictRequest,
  PredictResponse,
  PredictionPoint,
  TrafficApi,
} from '../types/traffic'

const DEFAULT_API_BASE = '/api'

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE
).replace(/\/$/, '')

function toUserMessage(status: number, serverMessage: string | null): string {
  if (status === 400) return serverMessage ?? 'Invalid request.'
  if (status === 500 || status === 503) {
    return serverMessage ?? 'Prediction failed.'
  }
  return serverMessage ?? `Request failed (${status}).`
}

type LocationsResponse = {
  locations?: Junction[]
}

type HistoryItem = {
  datetime?: string
  timestamp?: string
  vehicles?: number
  actual?: number
  congestion?: CongestionLevel
}

type HistoryResponse = {
  history?: HistoryItem[]
}

function isCongestion(value: unknown): value is CongestionLevel {
  return (
    value === 'clear' ||
    value === 'moderate' ||
    value === 'heavy' ||
    value === 'severe'
  )
}

function messageFromBody(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null
  const error = (body as { error?: unknown }).error
  return typeof error === 'string' && error.trim() ? error : null
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text) as unknown
  } catch {
    return null
  }
}

async function requestJson(path: string, init?: RequestInit): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    throw new Error(
      'Could not reach the prediction server. Start Flask on port 5000 (see README), then refresh.',
    )
  }

  const body = await readJson(response)
  if (!response.ok) {
    if (response.status === 502 || response.status === 504) {
      throw new Error(
        'Could not reach the prediction server. Start Flask on port 5000 (see README), then refresh.',
      )
    }
    throw new Error(toUserMessage(response.status, messageFromBody(body)))
  }
  return body
}

type RawPredictPoint = {
  timestamp?: string
  datetime?: string
  predicted?: number
  actual?: number | null
  congestion?: CongestionLevel
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function mapPredictPoint(item: RawPredictPoint): PredictionPoint {
  const timestamp = item.timestamp ?? item.datetime ?? ''
  const predicted = finiteNumber(item.predicted)
  if (!timestamp || predicted === undefined) {
    throw new Error('Prediction failed.')
  }
  const actual = finiteNumber(item.actual)
  return {
    timestamp,
    predicted,
    actual,
    congestion: isCongestion(item.congestion) ? item.congestion : 'clear',
  }
}

function parsePredictResponse(body: unknown): PredictResponse {
  if (!body || typeof body !== 'object') {
    throw new Error('Prediction failed.')
  }
  const raw = body as Record<string, unknown>
  if (!Array.isArray(raw.points)) {
    throw new Error('Prediction failed.')
  }
  if (typeof raw.model !== 'string' || raw.model.trim() === '') {
    throw new Error('Prediction failed.')
  }
  if (typeof raw.features !== 'number' || typeof raw.lookback !== 'number') {
    throw new Error('Prediction failed.')
  }
  if (typeof raw.historical_simulation !== 'boolean') {
    throw new Error('Prediction failed.')
  }
  const peakCongestion = raw.peakCongestion
  if (!isCongestion(peakCongestion)) {
    throw new Error('Prediction failed.')
  }
  return {
    locationId: String(raw.locationId ?? ''),
    start: String(raw.start ?? ''),
    end: String(raw.end ?? ''),
    points: raw.points.map((point) => mapPredictPoint(point as RawPredictPoint)),
    peakCongestion,
    peakVolume: finiteNumber(raw.peakVolume) ?? 0,
    model: raw.model,
    features: raw.features,
    lookback: raw.lookback,
    historical_simulation: raw.historical_simulation,
    junction: finiteNumber(raw.junction),
    predicted_vehicles: finiteNumber(raw.predicted_vehicles),
  }
}

function mapHistoryPoint(item: HistoryItem): PredictionPoint {
  const timestamp = item.timestamp ?? item.datetime ?? ''
  const actual = item.actual ?? item.vehicles
  const congestion = isCongestion(item.congestion) ? item.congestion : 'clear'
  const actualValue = typeof actual === 'number' && Number.isFinite(actual) ? actual : undefined
  return {
    timestamp,
    predicted: actualValue ?? 0,
    actual: actualValue,
    congestion,
  }
}

function toHourPayload(value: string): string {
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value)) {
    return `${value}:00`
  }
  return value
}

export const trafficApi: TrafficApi = {
  async getLocations() {
    const body = (await requestJson('/locations')) as LocationsResponse
    const locations = body.locations
    if (!Array.isArray(locations) || locations.length === 0) {
      throw new Error('Could not load junctions.')
    }
    return locations
  },

  async predict(request: PredictRequest): Promise<PredictResponse> {
    const body = await requestJson('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        locationId: request.locationId,
        start: toHourPayload(request.start),
        end: toHourPayload(request.end),
      }),
    })
    return parsePredictResponse(body)
  },

  async getHistory(locationId, start, end) {
    const params = new URLSearchParams({
      locationId,
      start: toHourPayload(start),
      end: toHourPayload(end),
    })
    const body = (await requestJson(`/history?${params.toString()}`)) as HistoryResponse
    if (!Array.isArray(body.history)) {
      throw new Error('Could not load history.')
    }
    return body.history.map(mapHistoryPoint)
  },
}
