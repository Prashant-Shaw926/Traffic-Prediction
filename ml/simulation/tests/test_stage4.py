"""Stage 4 integration tests. Predictions come from the real Spatial GRU service."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path

import pandas as pd

from backend.config import (
    SPATIAL_FEATURE_CONFIG_PATH,
    SPATIAL_FEATURE_SCALER_PATH,
    SPATIAL_GRU_MODEL_PATH,
    SPATIAL_TARGET_SCALER_PATH,
)
from backend.services.spatial_prediction_service import predict_request
from ml.simulation import (
    SimulationInputError,
    build_simulation_input,
    calculate_traffic_influence,
    discover_relationships,
    run_simulation,
)
from ml.simulation.data_provider import load_historical_traffic

JUNE_15 = "2017-06-15 10:00:00"
JUNE_25 = "2017-06-25 10:00:00"
ARTIFACTS = (
    SPATIAL_GRU_MODEL_PATH,
    SPATIAL_FEATURE_SCALER_PATH,
    SPATIAL_TARGET_SCALER_PATH,
    SPATIAL_FEATURE_CONFIG_PATH,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class Stage4SimulationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hashes_before = {path: _sha256(path) for path in ARTIFACTS}

    def _assert_integrated(self, stamp: str) -> None:
        result = run_simulation(stamp)
        end = pd.Timestamp(stamp)
        data = build_simulation_input(stamp)
        relationships = discover_relationships(data)
        influence = calculate_traffic_influence(data, relationships)
        history = load_historical_traffic()

        self.assertEqual(result.timestamp, end)
        self.assertEqual(result.metadata.window_end, end)
        self.assertEqual(result.metadata.window_start, data.window.start)
        self.assertTrue(result.metadata.historical_simulation)
        self.assertEqual(result.metadata.model, "Spatial_GRU")
        self.assertEqual(result.metadata.features, 53)
        self.assertEqual(result.metadata.lookback, 168)
        self.assertEqual(result.metadata.relationship_method, "lagged_pearson")
        self.assertEqual(result.metadata.influence_score_kind, "influence_index")
        self.assertEqual(result.metadata.dataset_end, "2017-06-30 23:00:00")
        self.assertEqual(len(result.junctions), 4)
        self.assertEqual(result.relationships, relationships)
        self.assertEqual(result.influence_result, influence)
        self.assertTrue(all(item.window_end == end for item in result.relationships))
        self.assertEqual(result.influence_result.simulation_timestamp, end)
        for state in data.window.states:
            self.assertTrue(all(item.timestamp <= end for item in state.observations))

        for junction_state in result.junctions:
            self.assertIn(junction_state.junction_id, (1, 2, 3, 4))
            direct = predict_request(
                {"junction": junction_state.junction_id, "datetime": end.isoformat()}
            )
            self.assertAlmostEqual(
                junction_state.predicted_traffic,
                float(direct["predicted_vehicles"]),
                places=6,
            )
            self.assertTrue(pd.notna(junction_state.predicted_traffic))
            self.assertNotAlmostEqual(
                junction_state.predicted_traffic,
                float(junction_state.incoming_influence),
            )
            matched = history.loc[
                (history["Junction"] == junction_state.junction_id)
                & (history["DateTime"] == end),
                "Vehicles",
            ]
            self.assertEqual(len(matched), 1)
            self.assertEqual(junction_state.historical_traffic, float(matched.iloc[0]))

    def test_june_15_integrates_prediction_relationships_and_influence(self) -> None:
        self._assert_integrated(JUNE_15)

    def test_june_25_integrates_prediction_relationships_and_influence(self) -> None:
        self._assert_integrated(JUNE_25)

    def test_same_timestamp_is_deterministic(self) -> None:
        first = run_simulation(JUNE_15)
        second = run_simulation(JUNE_15)
        self.assertEqual(first.relationships, second.relationships)
        self.assertEqual(first.influence_result, second.influence_result)
        self.assertEqual(first.metadata, second.metadata)
        for left, right in zip(first.junctions, second.junctions):
            self.assertEqual(left.junction_id, right.junction_id)
            self.assertAlmostEqual(left.predicted_traffic, right.predicted_traffic, places=6)
            self.assertEqual(left.historical_traffic, right.historical_traffic)
            self.assertEqual(left.incoming_influence, right.incoming_influence)

    def test_invalid_and_post_dataset_timestamps_are_rejected(self) -> None:
        for value in ("2017-07-01 10:00:00", "not-a-datetime"):
            with self.subTest(value=value):
                with self.assertRaises(SimulationInputError):
                    run_simulation(value)

    def test_spatial_artifacts_are_unchanged(self) -> None:
        for path, digest in self.hashes_before.items():
            self.assertEqual(_sha256(path), digest)


if __name__ == "__main__":
    unittest.main()
