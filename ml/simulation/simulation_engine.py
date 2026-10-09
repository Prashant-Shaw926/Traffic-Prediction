"""Orchestrate one historical traffic-network simulation.

Predicted traffic comes from the existing Spatial GRU service.
Relationship strength and the influence index come from Stages 2 and 3.
Those three outputs are not substituted for one another.
"""

from __future__ import annotations

import math

import pandas as pd

from backend.services.spatial_prediction_service import (
    load_spatial_artifacts,
    predict_request,
    spatial_model_loaded,
)
from ml.simulation.config import SimulationConfig, default_config
from ml.simulation.data_provider import build_simulation_input
from ml.simulation.models import (
    JunctionSimulationState,
    SimulationData,
    SimulationMetadata,
    TrafficInfluenceResult,
    TrafficNetworkSimulation,
)
from ml.simulation.network_relationship import discover_relationships
from ml.simulation.traffic_influence import calculate_traffic_influence

RELATIONSHIP_METHOD = "lagged_pearson"


def _historical_traffic(data: SimulationData, junction: int, timestamp: pd.Timestamp) -> float | None:
    for state in data.window.states:
        if state.junction != junction:
            continue
        for observation in state.observations:
            if pd.Timestamp(observation.timestamp) == timestamp:
                return float(observation.vehicles)
            if observation.timestamp > timestamp:
                raise AssertionError("Future observation in simulation window")
    return None


def _predict_junctions(data: SimulationData) -> tuple[dict[int, float], SimulationMetadata]:
    if not spatial_model_loaded():
        load_spatial_artifacts()
    timestamp = pd.Timestamp(data.request.timestamp)
    payload_time = timestamp.isoformat()
    predictions: dict[int, float] = {}
    model_name: str | None = None
    features: int | None = None
    lookback: int | None = None
    historical_simulation: bool | None = None
    for junction in data.config.supported_junctions:
        response = predict_request({"junction": int(junction), "datetime": payload_time})
        predicted = response.get("predicted_vehicles")
        if not isinstance(predicted, (int, float)) or not math.isfinite(float(predicted)):
            raise RuntimeError(f"Spatial GRU prediction failed for junction {junction}.")
        name = response.get("model")
        feature_count = response.get("features")
        lookback_hours = response.get("lookback")
        simulation_flag = response.get("historical_simulation")
        if model_name is None:
            model_name = str(name)
            features = int(feature_count)
            lookback = int(lookback_hours)
            historical_simulation = bool(simulation_flag)
        elif (name, feature_count, lookback_hours, simulation_flag) != (
            model_name,
            features,
            lookback,
            historical_simulation,
        ):
            raise RuntimeError("Spatial GRU metadata disagreed across junctions.")
        predictions[int(junction)] = float(predicted)
    if model_name is None or features is None or lookback is None or historical_simulation is None:
        raise RuntimeError("Spatial GRU prediction failed.")
    metadata = SimulationMetadata(
        historical_simulation=historical_simulation,
        model=model_name,
        features=features,
        lookback=lookback,
        relationship_method=RELATIONSHIP_METHOD,
        supported_lags=data.config.supported_lags,
        window_start=pd.Timestamp(data.window.start),
        window_end=pd.Timestamp(data.window.end),
        influence_score_kind="influence_index",
        dataset_end=data.config.dataset_end,
    )
    return predictions, metadata


def _junction_states(
    data: SimulationData,
    predictions: dict[int, float],
    influence: TrafficInfluenceResult,
) -> tuple[JunctionSimulationState, ...]:
    timestamp = pd.Timestamp(data.request.timestamp)
    by_influence = {item.junction_id: item for item in influence.junction_influences}
    states: list[JunctionSimulationState] = []
    for junction in data.config.supported_junctions:
        if junction not in predictions:
            raise RuntimeError(f"Spatial GRU prediction missing for junction {junction}.")
        junction_influence = by_influence.get(junction)
        if junction_influence is None:
            raise RuntimeError(f"Influence result missing for junction {junction}.")
        states.append(
            JunctionSimulationState(
                junction_id=int(junction),
                predicted_traffic=predictions[junction],
                historical_traffic=_historical_traffic(data, junction, timestamp),
                incoming_influence=junction_influence.incoming_influence,
                valid_source_count=junction_influence.valid_source_count,
            )
        )
    return tuple(states)


def run_simulation(
    timestamp: object,
    config: SimulationConfig | None = None,
) -> TrafficNetworkSimulation:
    """Run Stages 1-3 and Spatial GRU inference for one historical timestamp."""
    settings = config if config is not None else default_config()
    data = build_simulation_input(timestamp, settings)
    relationships = discover_relationships(data)
    influence = calculate_traffic_influence(data, relationships)
    predictions, metadata = _predict_junctions(data)
    return TrafficNetworkSimulation(
        timestamp=pd.Timestamp(data.request.timestamp),
        junctions=_junction_states(data, predictions, influence),
        relationships=relationships,
        influence_result=influence,
        metadata=metadata,
    )
