export type SimulationJunction = {
  junction_id: number
  predicted_traffic: number
  historical_traffic: number | null
  incoming_influence: number | null
  valid_source_count: number
}

export type SimulationRelationship = {
  source_junction: number
  target_junction: number
  lag_hours: number
  correlation: number
  absolute_correlation: number
  paired_observations: number
  window_start: string
  window_end: string
}

export type InfluenceContribution = {
  source_junction: number
  target_junction: number
  lag_hours: number
  signed_correlation: number
  relationship_weight: number
  source_timestamp: string
  source_traffic: number
  contribution: number
}

export type ExcludedContribution = {
  source_junction: number
  target_junction: number
  lag_hours: number
  signed_correlation: number
  relationship_weight: number
  source_timestamp: string
  reason: string
}

export type JunctionInfluence = {
  junction_id: number
  incoming_influence: number | null
  valid_source_count: number
  contributions: InfluenceContribution[]
}

export type SimulationResponse = {
  timestamp: string
  junctions: SimulationJunction[]
  relationships: SimulationRelationship[]
  influence_result: {
    simulation_timestamp: string
    metadata: {
      score_kind: string
      formula: string
      description: string
    }
    junction_influences: JunctionInfluence[]
    excluded_contributions: ExcludedContribution[]
  }
  metadata: {
    historical_simulation: boolean
    model: string
    features: number
    lookback: number
    relationship_method: string
    supported_lags: number[]
    window_start: string
    window_end: string
    influence_score_kind: string
    dataset_end: string
  }
}
