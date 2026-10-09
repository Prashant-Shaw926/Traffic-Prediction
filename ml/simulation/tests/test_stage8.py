"""Stage 8 synthetic checks for sign, lag selection, and excluded sources.

These fixtures are in memory. They do not change the dataset or production thresholds.
"""

from __future__ import annotations

import unittest

import pandas as pd

from ml.simulation import (
    SimulationConfig,
    calculate_traffic_influence,
    discover_relationships,
)
from ml.simulation.config import RelationshipSettings
from ml.simulation.models import (
    JunctionTrafficState,
    SimulationData,
    SimulationRequest,
    SimulationWindow,
    TrafficObservation,
)
from ml.simulation.network_relationship import aligned_pair_values, pearson_correlation
from ml.simulation.traffic_influence import MISSING_SOURCE_OBSERVATION

START = pd.Timestamp("2017-06-15 10:00:00")


def _hourly(values: list[float]) -> tuple[TrafficObservation, ...]:
    return tuple(
        TrafficObservation(
            timestamp=START + pd.Timedelta(hours=index),
            vehicles=float(value),
        )
        for index, value in enumerate(values)
    )


def _data(
    states: tuple[JunctionTrafficState, ...],
    lags: tuple[int, ...],
    minimum_pairs: int,
    junctions: tuple[int, ...],
) -> SimulationData:
    end = START + pd.Timedelta(hours=max(len(state.observations) for state in states) - 1)
    return SimulationData(
        request=SimulationRequest(timestamp=end),
        window=SimulationWindow(start=START, end=end, states=states),
        config=SimulationConfig(
            historical_window_hours=int((end - START).total_seconds() // 3600) + 1,
            supported_junctions=junctions,
            supported_lags=lags,
            minimum_observations=1,
            relationship=RelationshipSettings(minimum_paired_observations=minimum_pairs),
        ),
    )


def _pair(relationships, source: int, target: int):
    return next(
        item
        for item in relationships
        if item.source_junction == source and item.target_junction == target
    )


class Stage8SyntheticTests(unittest.TestCase):
    def test_negative_lag_wins_when_its_absolute_correlation_is_larger(self) -> None:
        source = JunctionTrafficState(junction=1, observations=_hourly([1, 100, 2, 90, 3, 80]))
        target = JunctionTrafficState(
            junction=2,
            observations=_hourly([0, -1, -100, -2, -90, -3]),
        )
        data = _data((source, target), (1, 2), minimum_pairs=4, junctions=(1, 2))
        selected = _pair(discover_relationships(data), 1, 2)
        lag2_pairs = aligned_pair_values(source, target, 2, data.window.end)
        lag2 = pearson_correlation(
            [item[0] for item in lag2_pairs],
            [item[1] for item in lag2_pairs],
        )
        self.assertIsNotNone(lag2)
        assert lag2 is not None
        self.assertEqual(selected.lag_hours, 1)
        self.assertLess(selected.correlation, 0.0)
        self.assertAlmostEqual(selected.absolute_correlation, abs(selected.correlation))
        self.assertAlmostEqual(selected.correlation, -1.0)
        self.assertGreater(selected.absolute_correlation, abs(lag2))

    def test_equal_absolute_correlation_keeps_the_earlier_lag(self) -> None:
        source = JunctionTrafficState(junction=1, observations=_hourly([1, 2, 3, 4, 5, 6]))
        target = JunctionTrafficState(
            junction=2,
            observations=(
                TrafficObservation(timestamp=START, vehicles=0.0),
                TrafficObservation(timestamp=START + pd.Timedelta(hours=2), vehicles=-2.0),
                TrafficObservation(timestamp=START + pd.Timedelta(hours=3), vehicles=-3.0),
                TrafficObservation(timestamp=START + pd.Timedelta(hours=4), vehicles=-4.0),
                TrafficObservation(timestamp=START + pd.Timedelta(hours=5), vehicles=-5.0),
            ),
        )
        data = _data((source, target), (1, 2), minimum_pairs=4, junctions=(1, 2))
        selected = _pair(discover_relationships(data), 1, 2)
        self.assertEqual(selected.lag_hours, 1)
        self.assertLess(selected.correlation, 0.0)
        self.assertAlmostEqual(selected.absolute_correlation, 1.0)
        self.assertAlmostEqual(selected.absolute_correlation, abs(selected.correlation))

    def test_positive_lag_wins_when_its_absolute_correlation_is_larger(self) -> None:
        source = JunctionTrafficState(junction=1, observations=_hourly([1, 100, 2, 90, 3, 80]))
        target = JunctionTrafficState(junction=2, observations=_hourly([0, 0, 1, 100, 2, 90]))
        data = _data((source, target), (1, 2), minimum_pairs=4, junctions=(1, 2))
        selected = _pair(discover_relationships(data), 1, 2)
        self.assertEqual(selected.lag_hours, 2)
        self.assertGreater(selected.correlation, 0.0)
        self.assertAlmostEqual(selected.absolute_correlation, abs(selected.correlation))
        self.assertAlmostEqual(selected.correlation, 1.0)

    def test_influence_weights_a_negative_correlation_by_its_absolute_value(self) -> None:
        source = JunctionTrafficState(junction=1, observations=_hourly([1, 100, 2, 90, 3, 80]))
        target = JunctionTrafficState(
            junction=2,
            observations=_hourly([0, -1, -100, -2, -90, -3]),
        )
        data = _data((source, target), (1,), minimum_pairs=4, junctions=(1, 2))
        relationships = discover_relationships(data)
        selected = _pair(relationships, 1, 2)
        result = calculate_traffic_influence(data, relationships)
        influences = {item.junction_id: item for item in result.junction_influences}
        contribution = influences[2].contributions[0]
        source_traffic = 3.0
        self.assertLess(selected.correlation, 0.0)
        self.assertEqual(contribution.lag_hours, selected.lag_hours)
        self.assertEqual(contribution.signed_correlation, selected.correlation)
        self.assertEqual(contribution.relationship_weight, selected.absolute_correlation)
        self.assertGreaterEqual(contribution.relationship_weight, 0.0)
        self.assertEqual(contribution.source_traffic, source_traffic)
        self.assertAlmostEqual(
            contribution.contribution,
            contribution.relationship_weight * source_traffic,
        )
        self.assertAlmostEqual(influences[2].incoming_influence, contribution.contribution)
        self.assertEqual(influences[2].valid_source_count, 1)

    def test_missing_source_hour_is_excluded_and_not_zero_filled(self) -> None:
        present = JunctionTrafficState(junction=1, observations=_hourly([1, 2, 3, 4, 5, 6]))
        target = JunctionTrafficState(junction=2, observations=_hourly([2, 4, 6, 8, 10, 12]))
        missing = JunctionTrafficState(junction=3, observations=_hourly([1, 3, 5, 7]))
        data = _data(
            (present, target, missing),
            (1,),
            minimum_pairs=3,
            junctions=(1, 2, 3, 4),
        )
        relationships = discover_relationships(data)
        result = calculate_traffic_influence(data, relationships)
        influences = {item.junction_id: item for item in result.junction_influences}
        included = influences[2].contributions
        self.assertEqual([item.source_junction for item in included], [1])
        self.assertEqual(included[0].source_traffic, 5.0)
        self.assertNotEqual(included[0].source_traffic, 0.0)
        self.assertAlmostEqual(
            influences[2].incoming_influence,
            sum(item.contribution for item in included),
        )
        excluded = [
            item
            for item in result.excluded_contributions
            if item.source_junction == 3 and item.target_junction == 2
        ]
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0].reason, MISSING_SOURCE_OBSERVATION)
        self.assertEqual(excluded[0].source_timestamp, data.window.end - pd.Timedelta(hours=1))
        self.assertAlmostEqual(excluded[0].relationship_weight, abs(excluded[0].signed_correlation))
        self.assertIsNone(influences[4].incoming_influence)
        self.assertEqual(influences[4].valid_source_count, 0)


if __name__ == "__main__":
    unittest.main()
