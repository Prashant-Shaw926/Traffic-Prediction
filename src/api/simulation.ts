import { API_BASE_URL } from './traffic'
import type {
  ExcludedContribution,
  InfluenceContribution,
  JunctionInfluence,
  SimulationJunction,
  SimulationRelationship,
  SimulationResponse,
} from '../types/simulation'

function messageFromBody(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null
  const error = (body as { error?: unknown }).error
  return typeof error === 'string' && error.trim() ? error : null
}

function toUserMessage(status: number, serverMessage: string | null): string {
  if (status === 400) return serverMessage ?? 'Invalid request.'
  if (status === 500 || status === 503) {
    return serverMessage ?? 'Simulation failed.'
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

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function requiredNullableNumber(raw: Record<string, unknown>, key: string): number | null {
  if (!(key in raw)) throw new Error(`Simulation failed: missing ${key}.`)
  const value = raw[key]
  if (value === null) return null
  const number = finiteNumber(value)
  if (number === undefined) throw new Error(`Simulation failed: missing ${key}.`)
  return number
}

function requireString(value: unknown, label: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`Simulation failed: missing ${label}.`)
  }
  return value
}

function requireNumber(value: unknown, label: string): number {
  const number = finiteNumber(value)
  if (number === undefined) throw new Error(`Simulation failed: missing ${label}.`)
  return number
}

function parseJunction(value: unknown): SimulationJunction {
  if (!value || typeof value !== 'object') throw new Error('Simulation failed.')
  const raw = value as Record<string, unknown>
  return {
    junction_id: requireNumber(raw.junction_id, 'junction_id'),
    predicted_traffic: requireNumber(raw.predicted_traffic, 'predicted_traffic'),
    historical_traffic: requiredNullableNumber(raw, 'historical_traffic'),
    incoming_influence: requiredNullableNumber(raw, 'incoming_influence'),
    valid_source_count: requireNumber(raw.valid_source_count, 'valid_source_count'),
  }
}

function parseRelationship(value: unknown): SimulationRelationship {
  if (!value || typeof value !== 'object') throw new Error('Simulation failed.')
  const raw = value as Record<string, unknown>
  return {
    source_junction: requireNumber(raw.source_junction, 'source_junction'),
    target_junction: requireNumber(raw.target_junction, 'target_junction'),
    lag_hours: requireNumber(raw.lag_hours, 'lag_hours'),
    correlation: requireNumber(raw.correlation, 'correlation'),
    absolute_correlation: requireNumber(raw.absolute_correlation, 'absolute_correlation'),
    paired_observations: requireNumber(raw.paired_observations, 'paired_observations'),
    window_start: requireString(raw.window_start, 'window_start'),
    window_end: requireString(raw.window_end, 'window_end'),
  }
}

function parseContribution(value: unknown): InfluenceContribution {
  if (!value || typeof value !== 'object') throw new Error('Simulation failed.')
  const raw = value as Record<string, unknown>
  return {
    source_junction: requireNumber(raw.source_junction, 'source_junction'),
    target_junction: requireNumber(raw.target_junction, 'target_junction'),
    lag_hours: requireNumber(raw.lag_hours, 'lag_hours'),
    signed_correlation: requireNumber(raw.signed_correlation, 'signed_correlation'),
    relationship_weight: requireNumber(raw.relationship_weight, 'relationship_weight'),
    source_timestamp: requireString(raw.source_timestamp, 'source_timestamp'),
    source_traffic: requireNumber(raw.source_traffic, 'source_traffic'),
    contribution: requireNumber(raw.contribution, 'contribution'),
  }
}

function parseExcluded(value: unknown): ExcludedContribution {
  if (!value || typeof value !== 'object') throw new Error('Simulation failed.')
  const raw = value as Record<string, unknown>
  return {
    source_junction: requireNumber(raw.source_junction, 'source_junction'),
    target_junction: requireNumber(raw.target_junction, 'target_junction'),
    lag_hours: requireNumber(raw.lag_hours, 'lag_hours'),
    signed_correlation: requireNumber(raw.signed_correlation, 'signed_correlation'),
    relationship_weight: requireNumber(raw.relationship_weight, 'relationship_weight'),
    source_timestamp: requireString(raw.source_timestamp, 'source_timestamp'),
    reason: requireString(raw.reason, 'reason'),
  }
}

function parseJunctionInfluence(value: unknown): JunctionInfluence {
  if (!value || typeof value !== 'object') throw new Error('Simulation failed.')
  const raw = value as Record<string, unknown>
  if (!Array.isArray(raw.contributions)) throw new Error('Simulation failed.')
  return {
    junction_id: requireNumber(raw.junction_id, 'junction_id'),
    incoming_influence: requiredNullableNumber(raw, 'incoming_influence'),
    valid_source_count: requireNumber(raw.valid_source_count, 'valid_source_count'),
    contributions: raw.contributions.map(parseContribution),
  }
}

export function parseSimulationResponse(body: unknown): SimulationResponse {
  if (!body || typeof body !== 'object') throw new Error('Simulation failed.')
  const raw = body as Record<string, unknown>
  if (!Array.isArray(raw.junctions) || !Array.isArray(raw.relationships)) {
    throw new Error('Simulation failed.')
  }
  const influence = raw.influence_result
  const metadata = raw.metadata
  if (!influence || typeof influence !== 'object' || !metadata || typeof metadata !== 'object') {
    throw new Error('Simulation failed.')
  }
  const influenceRaw = influence as Record<string, unknown>
  const influenceMeta = influenceRaw.metadata
  if (!influenceMeta || typeof influenceMeta !== 'object') throw new Error('Simulation failed.')
  const influenceMetaRaw = influenceMeta as Record<string, unknown>
  if (!Array.isArray(influenceRaw.junction_influences)) throw new Error('Simulation failed.')
  if (!Array.isArray(influenceRaw.excluded_contributions)) throw new Error('Simulation failed.')
  const meta = metadata as Record<string, unknown>
  if (!Array.isArray(meta.supported_lags)) throw new Error('Simulation failed.')
  if (typeof meta.historical_simulation !== 'boolean') throw new Error('Simulation failed.')
  return {
    timestamp: requireString(raw.timestamp, 'timestamp'),
    junctions: raw.junctions.map(parseJunction),
    relationships: raw.relationships.map(parseRelationship),
    influence_result: {
      simulation_timestamp: requireString(
        influenceRaw.simulation_timestamp,
        'simulation_timestamp',
      ),
      metadata: {
        score_kind: requireString(influenceMetaRaw.score_kind, 'score_kind'),
        formula: requireString(influenceMetaRaw.formula, 'formula'),
        description: requireString(influenceMetaRaw.description, 'description'),
      },
      junction_influences: influenceRaw.junction_influences.map(parseJunctionInfluence),
      excluded_contributions: influenceRaw.excluded_contributions.map(parseExcluded),
    },
    metadata: {
      historical_simulation: meta.historical_simulation,
      model: requireString(meta.model, 'model'),
      features: requireNumber(meta.features, 'features'),
      lookback: requireNumber(meta.lookback, 'lookback'),
      relationship_method: requireString(meta.relationship_method, 'relationship_method'),
      supported_lags: meta.supported_lags.map((lag) => requireNumber(lag, 'supported_lags')),
      window_start: requireString(meta.window_start, 'window_start'),
      window_end: requireString(meta.window_end, 'window_end'),
      influence_score_kind: requireString(meta.influence_score_kind, 'influence_score_kind'),
      dataset_end: requireString(meta.dataset_end, 'dataset_end'),
    },
  }
}

function toHourPayload(value: string): string {
  if (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(value)) return `${value}:00`
  return value
}

export async function runSimulation(
  datetime: string,
  options?: { exact?: boolean },
): Promise<SimulationResponse> {
  const payload = options?.exact ? datetime : toHourPayload(datetime)
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/simulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ datetime: payload }),
    })
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
  return parseSimulationResponse(body)
}
