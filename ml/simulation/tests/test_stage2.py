"""Stage 2 lagged-association tests. Dataset values are not invented."""

from __future__ import annotations

import unittest

import pandas as pd

from ml.simulation import (
    SimulationConfig,
    SimulationInputError,
    build_simulation_input,
    discover_relationships,
    discover_relationships_at,
    pearson_correlation,
)
from ml.simulation.config import RelationshipSettings
from ml.simulation.models import (
    JunctionTrafficState,
    SimulationData,
    SimulationRequest,
    SimulationWindow,
    TrafficObservation,
)
from ml.simulation.network_relationship import aligned_pair_values

JUNE_15 = "2017-06-15 10:00:00"
JUNE_25 = "2017-06-25 10:00:00"
CANDIDATE_LAGS = (1, 2, 3, 24, 168)
ORDERED_PAIRS = tuple(
    (source, target)
    for source in (1, 2, 3, 4)
    for target in (1, 2, 3, 4)
    if source != target
)


def _observation(timestamp: str, vehicles: float) -> TrafficObservation:
    return TrafficObservation(timestamp=pd.Timestamp(timestamp), vehicles=float(vehicles))


class Stage2RelationshipTests(unittest.TestCase):
    def test_june_15_discovers_ordered_pairs_inside_the_window(self) -> None:
        data = build_simulation_input(JUNE_15)
        relationships = discover_relationships(data)
        end = pd.Timestamp(JUNE_15)
        self.assertEqual(data.window.end, end)
        self.assertEqual(data.window.start, end - pd.Timedelta(hours=167))
        self.assertEqual(tuple(state.junction for state in data.window.states), (1, 2, 3, 4))

        for state in data.window.states:
            self.assertTrue(all(item.timestamp <= end for item in state.observations))

        self.assertEqual(len(relationships), len(ORDERED_PAIRS))
        self.assertEqual(
            tuple((item.source_junction, item.target_junction) for item in relationships),
            ORDERED_PAIRS,
        )
        for item in relationships:
            self.assertNotEqual(item.source_junction, item.target_junction)
            self.assertIn(item.lag_hours, CANDIDATE_LAGS)
            self.assertGreaterEqual(item.correlation, -1.0)
            self.assertLessEqual(item.correlation, 1.0)
            self.assertEqual(item.absolute_correlation, abs(item.correlation))
            self.assertGreaterEqual(item.paired_observations, data.config.relationship.minimum_paired_observations)
            self.assertEqual(item.window_start, data.window.start)
            self.assertEqual(item.window_end, end)
            self.assertLessEqual(item.window_end, end)

    def test_lag_168_has_no_pairs_inside_the_168_hour_window(self) -> None:
        data = build_simulation_input(JUNE_15)
        states = {state.junction: state for state in data.window.states}
        for source, target in ORDERED_PAIRS:
            pairs = aligned_pair_values(states[source], states[target], 168, data.window.end)
            self.assertEqual(len(pairs), 0)

    def test_june_25_uses_its_own_timestamp(self) -> None:
        earlier = discover_relationships_at(JUNE_15)
        later = discover_relationships_at(JUNE_25)
        self.assertTrue(all(item.window_end == pd.Timestamp(JUNE_15) for item in earlier))
        self.assertTrue(all(item.window_end == pd.Timestamp(JUNE_25) for item in later))
        self.assertNotEqual(earlier[0].window_end, later[0].window_end)
        self.assertEqual(len(later), len(ORDERED_PAIRS))

    def test_same_timestamp_is_deterministic(self) -> None:
        self.assertEqual(
            discover_relationships_at(JUNE_15),
            discover_relationships_at(JUNE_15),
        )

    def test_invalid_timestamps_are_still_rejected(self) -> None:
        for value in ("2017-07-01 10:00:00", "not-a-datetime"):
            with self.subTest(value=value):
                with self.assertRaises(SimulationInputError):
                    discover_relationships_at(value)

    def test_negative_correlation_is_preserved(self) -> None:
        correlation = pearson_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        self.assertIsNotNone(correlation)
        assert correlation is not None
        self.assertLess(correlation, 0.0)
        self.assertGreaterEqual(correlation, -1.0)

    def test_zero_variance_is_not_a_valid_relationship(self) -> None:
        self.assertIsNone(pearson_correlation([4.0, 4.0, 4.0], [1.0, 2.0, 3.0]))

    def test_missing_hours_are_skipped_instead_of_using_adjacent_rows(self) -> None:
        source = JunctionTrafficState(
            junction=1,
            observations=(
                _observation("2017-06-15 10:00:00", 4.0),
                _observation("2017-06-15 13:00:00", 8.0),
            ),
        )
        target = JunctionTrafficState(
            junction=2,
            observations=(
                _observation("2017-06-15 11:00:00", 1.0),
                _observation("2017-06-15 12:00:00", 100.0),
                _observation("2017-06-15 14:00:00", 2.0),
            ),
        )
        end = pd.Timestamp("2017-06-15 14:00:00")
        pairs = aligned_pair_values(source, target, 1, end)
        self.assertEqual(pairs, ((4.0, 1.0), (8.0, 2.0)))

        config = SimulationConfig(
            historical_window_hours=5,
            supported_junctions=(1, 2),
            supported_lags=(1,),
            minimum_observations=1,
            relationship=RelationshipSettings(minimum_paired_observations=2),
        )
        data = SimulationData(
            request=SimulationRequest(timestamp=end),
            window=SimulationWindow(
                start=pd.Timestamp("2017-06-15 10:00:00"),
                end=end,
                states=(source, target),
            ),
            config=config,
        )
        relationships = discover_relationships(data)
        self.assertEqual(len(relationships), 1)
        selected = relationships[0]
        self.assertEqual(selected.source_junction, 1)
        self.assertEqual(selected.target_junction, 2)
        self.assertEqual(selected.lag_hours, 1)
        self.assertEqual(selected.paired_observations, 2)
        self.assertAlmostEqual(selected.correlation, 1.0)
        self.assertAlmostEqual(selected.absolute_correlation, 1.0)


if __name__ == "__main__":
    unittest.main()
