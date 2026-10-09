"""Calculate a formula-based incoming influence index.

I_j(T) = sum(absolute_correlation(i -> j) * V_i(T - k_ij))

The score is a mathematical index from lagged statistical associations.
It is not a predicted vehicle count and it is not evidence of physical causation.
"""

from __future__ import annotations

import pandas as pd

from ml.simulation.config import SimulationConfig
from ml.simulation.data_provider import build_simulation_input
from ml.simulation.models import (
    ExcludedInfluenceContribution,
    InfluenceMetadata,
    JunctionInfluence,
    JunctionRelationship,
    SimulationData,
    TrafficInfluenceContribution,
    TrafficInfluenceResult,
)
from ml.simulation.network_relationship import discover_relationships

MISSING_SOURCE_OBSERVATION = "missing exact source observation"
SOURCE_EQUALS_TARGET = "source and target junction must differ"
WINDOW_MISMATCH = "relationship window does not end at the simulation timestamp"
FUTURE_SOURCE = "source timestamp is after the simulation timestamp"

INFLUENCE_METADATA = InfluenceMetadata(
    score_kind="influence_index",
    formula="I_j(T) = sum(absolute_correlation * V_source(T - lag_hours))",
    description=(
        "Mathematical influence index from lagged statistical associations. "
        "Not a vehicle count and not a causal effect."
    ),
)


def _traffic_index(data: SimulationData) -> dict[int, dict[pd.Timestamp, float]]:
    indexed: dict[int, dict[pd.Timestamp, float]] = {}
    for state in data.window.states:
        by_time: dict[pd.Timestamp, float] = {}
        for observation in state.observations:
            if observation.timestamp > data.window.end:
                raise AssertionError("Future observation in simulation window")
            by_time[pd.Timestamp(observation.timestamp)] = float(observation.vehicles)
        indexed[state.junction] = by_time
    return indexed


def _exclude(
    relationship: JunctionRelationship,
    source_timestamp: pd.Timestamp,
    reason: str,
) -> ExcludedInfluenceContribution:
    return ExcludedInfluenceContribution(
        source_junction=relationship.source_junction,
        target_junction=relationship.target_junction,
        lag_hours=relationship.lag_hours,
        signed_correlation=relationship.correlation,
        relationship_weight=relationship.absolute_correlation,
        source_timestamp=source_timestamp,
        reason=reason,
    )


def calculate_traffic_influence(
    data: SimulationData,
    relationships: tuple[JunctionRelationship, ...],
) -> TrafficInfluenceResult:
    """Sum exact lagged source observations weighted by absolute correlation.

    A missing hour is omitted and reported. It is not interpolated or replaced with zero.
    """
    timestamp = pd.Timestamp(data.request.timestamp)
    traffic = _traffic_index(data)
    contributions: list[TrafficInfluenceContribution] = []
    excluded: list[ExcludedInfluenceContribution] = []

    for relationship in relationships:
        source_timestamp = timestamp - pd.Timedelta(hours=relationship.lag_hours)
        if relationship.source_junction == relationship.target_junction:
            excluded.append(_exclude(relationship, source_timestamp, SOURCE_EQUALS_TARGET))
            continue
        if pd.Timestamp(relationship.window_end) != timestamp:
            excluded.append(_exclude(relationship, source_timestamp, WINDOW_MISMATCH))
            continue
        if source_timestamp > timestamp:
            excluded.append(_exclude(relationship, source_timestamp, FUTURE_SOURCE))
            continue
        source_series = traffic.get(relationship.source_junction, {})
        if source_timestamp not in source_series:
            excluded.append(_exclude(relationship, source_timestamp, MISSING_SOURCE_OBSERVATION))
            continue
        source_traffic = source_series[source_timestamp]
        weight = float(relationship.absolute_correlation)
        contributions.append(
            TrafficInfluenceContribution(
                source_junction=relationship.source_junction,
                target_junction=relationship.target_junction,
                lag_hours=relationship.lag_hours,
                signed_correlation=float(relationship.correlation),
                relationship_weight=weight,
                source_timestamp=source_timestamp,
                source_traffic=source_traffic,
                contribution=weight * source_traffic,
            )
        )

    by_target: dict[int, list[TrafficInfluenceContribution]] = {
        junction: [] for junction in data.config.supported_junctions
    }
    for contribution in contributions:
        by_target.setdefault(contribution.target_junction, []).append(contribution)

    influences: list[JunctionInfluence] = []
    for junction in data.config.supported_junctions:
        items = tuple(sorted(by_target.get(junction, []), key=lambda item: item.source_junction))
        total = None if not items else float(sum(item.contribution for item in items))
        influences.append(
            JunctionInfluence(
                junction_id=junction,
                incoming_influence=total,
                valid_source_count=len(items),
                contributions=items,
            )
        )

    excluded.sort(key=lambda item: (item.target_junction, item.source_junction, item.lag_hours))
    return TrafficInfluenceResult(
        simulation_timestamp=timestamp,
        junction_influences=tuple(influences),
        excluded_contributions=tuple(excluded),
        metadata=INFLUENCE_METADATA,
    )


def calculate_traffic_influence_at(
    timestamp: object,
    config: SimulationConfig | None = None,
) -> TrafficInfluenceResult:
    """Validate T, rediscover relationships for that window, then apply the formula."""
    data = build_simulation_input(timestamp, config)
    return calculate_traffic_influence(data, discover_relationships(data))
