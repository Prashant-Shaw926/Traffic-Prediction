"""Stage 3 influence-index tests. Missing hours are not invented."""

from __future__ import annotations

import unittest

import pandas as pd

from ml.simulation import (
    SimulationConfig,
    SimulationInputError,
    calculate_traffic_influence,
    calculate_traffic_influence_at,
)
from ml.simulation.config import RelationshipSettings
from ml.simulation.data_provider import load_historical_traffic
from ml.simulation.models import (
    JunctionRelationship,
    JunctionTrafficState,
    SimulationData,
    SimulationRequest,
    SimulationWindow,
    TrafficObservation,
)
from ml.simulation.traffic_influence import MISSING_SOURCE_OBSERVATION

JUNE_15 = "2017-06-15 10:00:00"
JUNE_25 = "2017-06-25 10:00:00"


class Stage3InfluenceTests(unittest.TestCase):
    def _assert_real_timestamp(self, stamp: str) -> None:
        result = calculate_traffic_influence_at(stamp)
        end = pd.Timestamp(stamp)
        history = load_historical_traffic()
        self.assertEqual(result.simulation_timestamp, end)
        self.assertEqual(result.metadata.score_kind, "influence_index")
        self.assertIn("Not a vehicle count", result.metadata.description)
        self.assertEqual(
            tuple(item.junction_id for item in result.junction_influences),
            (1, 2, 3, 4),
        )
        self.assertEqual(len(result.excluded_contributions), 0)

        for junction in result.junction_influences:
            self.assertIsNotNone(junction.incoming_influence)
            self.assertEqual(junction.valid_source_count, len(junction.contributions))
            self.assertGreater(junction.valid_source_count, 0)
            total = sum(item.contribution for item in junction.contributions)
            self.assertAlmostEqual(junction.incoming_influence, total, places=9)
            target_at_t = history.loc[
                (history["Junction"] == junction.junction_id) & (history["DateTime"] == end),
                "Vehicles",
            ]
            self.assertEqual(len(target_at_t), 1)
            self.assertNotAlmostEqual(float(junction.incoming_influence), float(target_at_t.iloc[0]))
            for contribution in junction.contributions:
                self.assertNotEqual(contribution.source_junction, contribution.target_junction)
                self.assertEqual(contribution.target_junction, junction.junction_id)
                expected_time = end - pd.Timedelta(hours=contribution.lag_hours)
                self.assertEqual(contribution.source_timestamp, expected_time)
                self.assertLessEqual(contribution.source_timestamp, end)
                matched = history.loc[
                    (history["Junction"] == contribution.source_junction)
                    & (history["DateTime"] == expected_time),
                    "Vehicles",
                ]
                self.assertEqual(len(matched), 1)
                self.assertEqual(contribution.source_traffic, float(matched.iloc[0]))
                self.assertAlmostEqual(
                    contribution.contribution,
                    contribution.relationship_weight * contribution.source_traffic,
                )
                self.assertEqual(
                    contribution.relationship_weight,
                    abs(contribution.signed_correlation),
                )

    def test_june_15_influence_uses_exact_lagged_observations(self) -> None:
        self._assert_real_timestamp(JUNE_15)

    def test_june_25_influence_uses_exact_lagged_observations(self) -> None:
        self._assert_real_timestamp(JUNE_25)

    def test_same_timestamp_is_deterministic(self) -> None:
        self.assertEqual(
            calculate_traffic_influence_at(JUNE_15),
            calculate_traffic_influence_at(JUNE_15),
        )

    def test_invalid_timestamps_are_still_rejected(self) -> None:
        for value in ("2017-07-01 10:00:00", "not-a-datetime"):
            with self.subTest(value=value):
                with self.assertRaises(SimulationInputError):
                    calculate_traffic_influence_at(value)

    def test_missing_exact_source_hour_is_omitted_without_zero_fill(self) -> None:
        end = pd.Timestamp("2017-06-15 10:00:00")
        missing_time = end - pd.Timedelta(hours=1)
        present = TrafficObservation(timestamp=missing_time, vehicles=10.0)
        source = JunctionTrafficState(junction=1, observations=(present,))
        other = JunctionTrafficState(junction=3, observations=())
        target = JunctionTrafficState(
            junction=2,
            observations=(TrafficObservation(timestamp=end, vehicles=90.0),),
        )
        config = SimulationConfig(
            supported_junctions=(1, 2, 3),
            relationship=RelationshipSettings(minimum_paired_observations=1),
        )
        data = SimulationData(
            request=SimulationRequest(timestamp=end),
            window=SimulationWindow(start=missing_time, end=end, states=(source, target, other)),
            config=config,
        )
        included = JunctionRelationship(
            source_junction=1,
            target_junction=2,
            lag_hours=1,
            correlation=0.5,
            absolute_correlation=0.5,
            paired_observations=2,
            window_start=missing_time,
            window_end=end,
        )
        missing = JunctionRelationship(
            source_junction=3,
            target_junction=2,
            lag_hours=1,
            correlation=-0.25,
            absolute_correlation=0.25,
            paired_observations=2,
            window_start=missing_time,
            window_end=end,
        )
        result = calculate_traffic_influence(data, (included, missing))
        influences = {item.junction_id: item for item in result.junction_influences}
        self.assertAlmostEqual(influences[2].incoming_influence, 5.0)
        self.assertEqual(influences[2].valid_source_count, 1)
        self.assertEqual(influences[2].contributions[0].source_traffic, 10.0)
        self.assertEqual(len(result.excluded_contributions), 1)
        excluded = result.excluded_contributions[0]
        self.assertEqual(excluded.source_junction, 3)
        self.assertEqual(excluded.source_timestamp, missing_time)
        self.assertEqual(excluded.reason, MISSING_SOURCE_OBSERVATION)
        self.assertEqual(excluded.signed_correlation, -0.25)
        self.assertIsNone(influences[1].incoming_influence)
        self.assertIsNone(influences[3].incoming_influence)


if __name__ == "__main__":
    unittest.main()
