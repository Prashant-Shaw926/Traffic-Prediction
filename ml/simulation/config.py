"""Simulation settings. These are parameters, not junction-to-junction results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationshipSettings:
    """Reserved for a later relationship stage. Stage 1 does not compute weights."""

    minimum_paired_observations: int = 24


@dataclass(frozen=True)
class SimulationConfig:
    """Inputs that define a historical simulation window.

    Junction membership is a list of series to require. It is not a road graph
    and it does not assign influence between junctions.
    """

    historical_window_hours: int = 168
    supported_junctions: tuple[int, ...] = (1, 2, 3, 4)
    supported_lags: tuple[int, ...] = (1, 2, 3, 24, 168)
    minimum_observations: int = 24
    dataset_end: str = "2017-06-30 23:00:00"
    relationship: RelationshipSettings = RelationshipSettings()


def default_config() -> SimulationConfig:
    return SimulationConfig()
