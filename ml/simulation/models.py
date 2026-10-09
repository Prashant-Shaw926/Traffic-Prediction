"""Typed inputs for the historical traffic-network simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml.simulation.config import SimulationConfig


@dataclass(frozen=True)
class SimulationRequest:
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class TrafficObservation:
    timestamp: pd.Timestamp
    vehicles: float


@dataclass(frozen=True)
class JunctionTrafficState:
    junction: int
    observations: tuple[TrafficObservation, ...]


@dataclass(frozen=True)
class SimulationWindow:
    start: pd.Timestamp
    end: pd.Timestamp
    states: tuple[JunctionTrafficState, ...]


@dataclass(frozen=True)
class SimulationData:
    request: SimulationRequest
    window: SimulationWindow
    config: SimulationConfig


@dataclass(frozen=True)
class JunctionRelationship:
    """A lagged statistical association. This is not a road link or a causal claim."""

    source_junction: int
    target_junction: int
    lag_hours: int
    correlation: float
    absolute_correlation: float
    paired_observations: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp


@dataclass(frozen=True)
class TrafficInfluenceContribution:
    """One term in the influence index. Not a vehicle count and not a causal effect."""

    source_junction: int
    target_junction: int
    lag_hours: int
    signed_correlation: float
    relationship_weight: float
    source_timestamp: pd.Timestamp
    source_traffic: float
    contribution: float


@dataclass(frozen=True)
class ExcludedInfluenceContribution:
    """A selected relationship whose exact lagged source observation is unavailable."""

    source_junction: int
    target_junction: int
    lag_hours: int
    signed_correlation: float
    relationship_weight: float
    source_timestamp: pd.Timestamp
    reason: str


@dataclass(frozen=True)
class JunctionInfluence:
    junction_id: int
    incoming_influence: float | None
    valid_source_count: int
    contributions: tuple[TrafficInfluenceContribution, ...]


@dataclass(frozen=True)
class InfluenceMetadata:
    score_kind: str
    formula: str
    description: str


@dataclass(frozen=True)
class TrafficInfluenceResult:
    simulation_timestamp: pd.Timestamp
    junction_influences: tuple[JunctionInfluence, ...]
    excluded_contributions: tuple[ExcludedInfluenceContribution, ...]
    metadata: InfluenceMetadata


@dataclass(frozen=True)
class JunctionSimulationState:
    """Predicted traffic, observed traffic, and the influence index stay separate."""

    junction_id: int
    predicted_traffic: float
    historical_traffic: float | None
    incoming_influence: float | None
    valid_source_count: int


@dataclass(frozen=True)
class SimulationMetadata:
    historical_simulation: bool
    model: str
    features: int
    lookback: int
    relationship_method: str
    supported_lags: tuple[int, ...]
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    influence_score_kind: str
    dataset_end: str


@dataclass(frozen=True)
class TrafficNetworkSimulation:
    timestamp: pd.Timestamp
    junctions: tuple[JunctionSimulationState, ...]
    relationships: tuple[JunctionRelationship, ...]
    influence_result: TrafficInfluenceResult
    metadata: SimulationMetadata
