"""Serialize a historical simulation for POST /api/simulation.

This module does not calculate relationships or influence. It calls run_simulation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backend.utils.responses import ApiError, isoformat
from ml.simulation.data_provider import SimulationInputError
from ml.simulation.models import (
    ExcludedInfluenceContribution,
    JunctionInfluence,
    JunctionRelationship,
    JunctionSimulationState,
    TrafficInfluenceContribution,
    TrafficInfluenceResult,
    TrafficNetworkSimulation,
)
from ml.simulation.simulation_engine import run_simulation


def _stamp(value: pd.Timestamp) -> str:
    return isoformat(value)


def _junction_state(state: JunctionSimulationState) -> dict[str, Any]:
    return {
        "junction_id": state.junction_id,
        "predicted_traffic": state.predicted_traffic,
        "historical_traffic": state.historical_traffic,
        "incoming_influence": state.incoming_influence,
        "valid_source_count": state.valid_source_count,
    }


def _relationship(item: JunctionRelationship) -> dict[str, Any]:
    return {
        "source_junction": item.source_junction,
        "target_junction": item.target_junction,
        "lag_hours": item.lag_hours,
        "correlation": item.correlation,
        "absolute_correlation": item.absolute_correlation,
        "paired_observations": item.paired_observations,
        "window_start": _stamp(item.window_start),
        "window_end": _stamp(item.window_end),
    }


def _contribution(item: TrafficInfluenceContribution) -> dict[str, Any]:
    return {
        "source_junction": item.source_junction,
        "target_junction": item.target_junction,
        "lag_hours": item.lag_hours,
        "signed_correlation": item.signed_correlation,
        "relationship_weight": item.relationship_weight,
        "source_timestamp": _stamp(item.source_timestamp),
        "source_traffic": item.source_traffic,
        "contribution": item.contribution,
    }


def _excluded(item: ExcludedInfluenceContribution) -> dict[str, Any]:
    return {
        "source_junction": item.source_junction,
        "target_junction": item.target_junction,
        "lag_hours": item.lag_hours,
        "signed_correlation": item.signed_correlation,
        "relationship_weight": item.relationship_weight,
        "source_timestamp": _stamp(item.source_timestamp),
        "reason": item.reason,
    }


def _junction_influence(item: JunctionInfluence) -> dict[str, Any]:
    return {
        "junction_id": item.junction_id,
        "incoming_influence": item.incoming_influence,
        "valid_source_count": item.valid_source_count,
        "contributions": [_contribution(contribution) for contribution in item.contributions],
    }


def _influence(result: TrafficInfluenceResult) -> dict[str, Any]:
    return {
        "simulation_timestamp": _stamp(result.simulation_timestamp),
        "metadata": {
            "score_kind": result.metadata.score_kind,
            "formula": result.metadata.formula,
            "description": result.metadata.description,
        },
        "junction_influences": [
            _junction_influence(item) for item in result.junction_influences
        ],
        "excluded_contributions": [_excluded(item) for item in result.excluded_contributions],
    }


def serialize_simulation(result: TrafficNetworkSimulation) -> dict[str, Any]:
    metadata = result.metadata
    return {
        "timestamp": _stamp(result.timestamp),
        "junctions": [_junction_state(item) for item in result.junctions],
        "relationships": [_relationship(item) for item in result.relationships],
        "influence_result": _influence(result.influence_result),
        "metadata": {
            "historical_simulation": metadata.historical_simulation,
            "model": metadata.model,
            "features": metadata.features,
            "lookback": metadata.lookback,
            "relationship_method": metadata.relationship_method,
            "supported_lags": list(metadata.supported_lags),
            "window_start": _stamp(metadata.window_start),
            "window_end": _stamp(metadata.window_end),
            "influence_score_kind": metadata.influence_score_kind,
            "dataset_end": metadata.dataset_end,
        },
    }


def simulation_response(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("datetime")
    if raw is None or str(raw).strip() == "":
        raise ApiError("Missing request fields: datetime is required.", 400)
    try:
        result = run_simulation(raw)
    except SimulationInputError as exc:
        raise ApiError(str(exc), 400) from exc
    return serialize_simulation(result)
