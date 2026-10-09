"""Discover lagged statistical associations between junctions.

Relationships are recalculated from the Stage 1 window for the requested
timestamp. They are lagged associations, not road links and not causal effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.simulation.config import SimulationConfig
from ml.simulation.data_provider import build_simulation_input
from ml.simulation.models import (
    JunctionRelationship,
    JunctionTrafficState,
    SimulationData,
    TrafficObservation,
)


def pearson_correlation(left: list[float] | tuple[float, ...], right: list[float] | tuple[float, ...]) -> float | None:
    """Return a finite Pearson correlation, or None when it cannot be computed.

    A constant series has zero variance. That lag is invalid. It is not stored
    as NaN and it is not replaced with zero.
    """
    if len(left) < 2 or len(left) != len(right):
        return None
    source = np.asarray(left, dtype=np.float64)
    target = np.asarray(right, dtype=np.float64)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        return None
    if float(np.std(source)) == 0.0 or float(np.std(target)) == 0.0:
        return None
    value = float(np.corrcoef(source, target)[0, 1])
    if not np.isfinite(value):
        return None
    if value < -1.0 or value > 1.0:
        if abs(abs(value) - 1.0) <= 1e-8:
            return float(np.clip(value, -1.0, 1.0))
        return None
    return value


def _observations_by_timestamp(
    observations: tuple[TrafficObservation, ...],
    window_end: pd.Timestamp,
) -> dict[pd.Timestamp, float]:
    indexed: dict[pd.Timestamp, float] = {}
    for observation in observations:
        if observation.timestamp > window_end:
            raise AssertionError("Future observation in simulation window")
        indexed[pd.Timestamp(observation.timestamp)] = float(observation.vehicles)
    return indexed


def aligned_pair_values(
    source: JunctionTrafficState,
    target: JunctionTrafficState,
    lag_hours: int,
    window_end: pd.Timestamp,
) -> tuple[tuple[float, float], ...]:
    """Pair V_source(t-k) with V_target(t) by timestamp.

    Adjacent rows are not treated as adjacent hours. A missing hour is skipped.
    """
    if lag_hours < 1:
        raise ValueError("lag_hours must be at least 1")
    source_at = _observations_by_timestamp(source.observations, window_end)
    pairs: list[tuple[float, float]] = []
    for observation in target.observations:
        if observation.timestamp > window_end:
            raise AssertionError("Future target observation")
        source_time = pd.Timestamp(observation.timestamp) - pd.Timedelta(hours=lag_hours)
        if source_time > window_end:
            raise AssertionError("Future source timestamp")
        source_vehicles = source_at.get(source_time)
        if source_vehicles is None:
            continue
        pairs.append((source_vehicles, float(observation.vehicles)))
    return tuple(pairs)


def _select_relationship(
    source: JunctionTrafficState,
    target: JunctionTrafficState,
    data: SimulationData,
) -> JunctionRelationship | None:
    config: SimulationConfig = data.config
    minimum = config.relationship.minimum_paired_observations
    selected: JunctionRelationship | None = None
    for lag in config.supported_lags:
        pairs = aligned_pair_values(source, target, int(lag), data.window.end)
        if len(pairs) < minimum:
            continue
        correlation = pearson_correlation(
            [item[0] for item in pairs],
            [item[1] for item in pairs],
        )
        if correlation is None:
            continue
        candidate = JunctionRelationship(
            source_junction=source.junction,
            target_junction=target.junction,
            lag_hours=int(lag),
            correlation=correlation,
            absolute_correlation=abs(correlation),
            paired_observations=len(pairs),
            window_start=pd.Timestamp(data.window.start),
            window_end=pd.Timestamp(data.window.end),
        )
        if selected is None or candidate.absolute_correlation > selected.absolute_correlation:
            selected = candidate
    return selected


def discover_relationships(data: SimulationData) -> tuple[JunctionRelationship, ...]:
    """Select one lagged association per ordered pair that has a valid lag.

    Absolute correlation chooses the lag. The signed correlation is kept.
    There is no strength threshold and no stored edge list.
    """
    states = {state.junction: state for state in data.window.states}
    relationships: list[JunctionRelationship] = []
    for source_id in states:
        for target_id in states:
            if source_id == target_id:
                continue
            selected = _select_relationship(states[source_id], states[target_id], data)
            if selected is not None:
                relationships.append(selected)
    relationships.sort(key=lambda item: (item.source_junction, item.target_junction))
    return tuple(relationships)


def discover_relationships_at(
    timestamp: object,
    config: SimulationConfig | None = None,
) -> tuple[JunctionRelationship, ...]:
    """Build a fresh Stage 1 window at `timestamp` and discover relationships."""
    return discover_relationships(build_simulation_input(timestamp, config))
