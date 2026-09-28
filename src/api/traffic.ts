import type {
  CongestionLevel,
  Junction,
  PredictRequest,
  PredictResponse,
  PredictionPoint,
  TrafficApi,
} from '../types/traffic'

const DEFAULT_API_BASE = '/api'
const INSUFFICIENT_HISTORY =
  'Insufficient historical data for the requested prediction timestamp.'
const INSUFFICIENT_USER =
  'Prediction unavailable for this time because sufficient historical data is not available.'

export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE
).replace(/\/$/, '')

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

function toUserMessage(status: number, serverMessage: string | null): string {
  if (serverMessage === INSUFFICIENT_HISTORY) return INSUFFICIENT_USER
  if (status === 400) return serverMessage ?? 'Invalid request.'
  if (status === 500 || status === 503) {
    return serverMessage ?? 'Prediction failed.'
  }
  return serverMessage ?? `Request failed (${status}).`
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

function mapHistoryPoint(item: HistoryItem): PredictionPoint {
  const timestamp = item.timestamp ?? item.datetime ?? ''
  const actual = item.actual ?? item.vehicles
  const congestion = isCongestion(item.congestion) ? item.congestion : 'clear'
  return {
    timestamp,
    predicted: typeof actual === 'number' ? actual : 0,
    actual: typeof actual === 'number' ? actual : undefined,
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
    const body = (await requestJson('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        locationId: request.locationId,
        start: toHourPayload(request.start),
        end: toHourPayload(request.end),
      }),
    })) as PredictResponse

    if (!body || !Array.isArray(body.points)) {
      throw new Error('Prediction failed.')
    }
    return body
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
