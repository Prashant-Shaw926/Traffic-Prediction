"""Stage 7 consistency checks. These tests do not change the simulation formula."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import pandas as pd

from backend.config import (
    FEATURE_SCALER_PATH,
    GRU_MODEL_PATH,
    SPATIAL_FEATURE_CONFIG_PATH,
    SPATIAL_FEATURE_SCALER_PATH,
    SPATIAL_GRU_MODEL_PATH,
    SPATIAL_TARGET_SCALER_PATH,
    TARGET_SCALER_PATH,
)
from backend.services.spatial_prediction_service import predict_request
from ml.simulation import (
    build_simulation_input,
    calculate_traffic_influence,
    discover_relationships,
    run_simulation,
)
from ml.simulation.config import default_config
from ml.simulation.data_provider import load_historical_traffic
from ml.simulation.traffic_influence import MISSING_SOURCE_OBSERVATION

STAMPS = (
    "2017-06-15 10:00:00",
    "2017-06-25 10:00:00",
    "2017-06-15 18:00:00",
    "2017-06-20 08:00:00",
)
REPEAT_STAMPS = STAMPS[:2]
ARTIFACTS = (
    SPATIAL_GRU_MODEL_PATH,
    SPATIAL_FEATURE_SCALER_PATH,
    SPATIAL_TARGET_SCALER_PATH,
    SPATIAL_FEATURE_CONFIG_PATH,
    GRU_MODEL_PATH,
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _signature(result) -> tuple:
    return tuple(
        (
            item.source_junction,
            item.target_junction,
            item.lag_hours,
            item.correlation,
            item.absolute_correlation,
            item.paired_observations,
        )
        for item in result.relationships
    )


def _lookup(history: pd.DataFrame) -> dict[tuple[int, pd.Timestamp], float]:
    values: dict[tuple[int, pd.Timestamp], float] = {}
    for row in history.itertuples(index=False):
        values[(int(row.Junction), pd.Timestamp(row.DateTime))] = float(row.Vehicles)
    return values


class Stage7ConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hashes_before = {path: _sha256(path) for path in ARTIFACTS}
        cls.results = {stamp: run_simulation(stamp) for stamp in STAMPS}
        cls.history = _lookup(load_historical_traffic())
        cls.config = default_config()

    def test_each_timestamp_matches_a_fresh_discovery(self) -> None:
        signatures = {}
        for stamp, result in self.results.items():
            with self.subTest(stamp=stamp):
                data = build_simulation_input(stamp)
                relationships = discover_relationships(data)
                influence = calculate_traffic_influence(data, relationships)
                self.assertEqual(result.relationships, relationships)
                self.assertEqual(result.influence_result, influence)
                self.assertEqual(result.metadata.relationship_method, "lagged_pearson")
                self.assertEqual(result.metadata.influence_score_kind, "influence_index")
                self.assertEqual(tuple(result.metadata.supported_lags), self.config.supported_lags)
                signatures[stamp] = _signature(result)
        unique = len(set(signatures.values()))
        print(f"STAGE7_UNIQUE_RELATIONSHIP_SIGNATURES {unique} of {len(signatures)}")
        for left in STAMPS:
            for right in STAMPS:
                if left >= right:
                    continue
                same = signatures[left] == signatures[right]
                print(f"STAGE7_SIGNATURE_EQUAL {left} {right} {same}")

    def test_influence_arithmetic_uses_raw_lagged_traffic(self) -> None:
        for stamp, result in self.results.items():
            with self.subTest(stamp=stamp):
                by_pair = {
                    (item.source_junction, item.target_junction): item
                    for item in result.relationships
                }
                included_ids = set()
                for junction in result.influence_result.junction_influences:
                    contributions = junction.contributions
                    if not contributions:
                        self.assertIsNone(junction.incoming_influence)
                        self.assertEqual(junction.valid_source_count, 0)
                        continue
                    total = 0.0
                    for item in contributions:
                        relationship = by_pair[(item.source_junction, item.target_junction)]
                        source_time = pd.Timestamp(result.timestamp) - pd.Timedelta(
                            hours=item.lag_hours
                        )
                        self.assertEqual(item.lag_hours, relationship.lag_hours)
                        self.assertEqual(item.signed_correlation, relationship.correlation)
                        self.assertEqual(item.relationship_weight, relationship.absolute_correlation)
                        self.assertEqual(pd.Timestamp(item.source_timestamp), source_time)
                        observed = self.history[(item.source_junction, source_time)]
                        self.assertEqual(item.source_traffic, observed)
                        self.assertTrue(math.isfinite(item.contribution))
                        self.assertAlmostEqual(
                            item.contribution,
                            item.relationship_weight * item.source_traffic,
                        )
                        total += item.contribution
                        included_ids.add((item.source_junction, item.target_junction, item.lag_hours))
                    self.assertEqual(junction.valid_source_count, len(contributions))
                    self.assertAlmostEqual(junction.incoming_influence, total)
                for excluded in result.influence_result.excluded_contributions:
                    key = (
                        excluded.source_junction,
                        excluded.target_junction,
                        excluded.lag_hours,
                    )
                    self.assertNotIn(key, included_ids)
                    if excluded.reason == MISSING_SOURCE_OBSERVATION:
                        self.assertNotIn(
                            (excluded.source_junction, pd.Timestamp(excluded.source_timestamp)),
                            self.history,
                        )

    def test_selected_correlation_keeps_its_sign(self) -> None:
        for stamp, result in self.results.items():
            with self.subTest(stamp=stamp):
                self.assertGreaterEqual(len(result.relationships), 1)
                for item in result.relationships:
                    self.assertIn(item.lag_hours, self.config.supported_lags)
                    self.assertGreaterEqual(
                        item.paired_observations,
                        self.config.relationship.minimum_paired_observations,
                    )
                    self.assertAlmostEqual(item.absolute_correlation, abs(item.correlation))
                    self.assertGreaterEqual(item.absolute_correlation, 0.0)
                    self.assertLessEqual(item.correlation, 1.0)
                    self.assertGreaterEqual(item.correlation, -1.0)

    def test_predictions_match_spatial_service_and_stay_separate(self) -> None:
        features = json.loads(SPATIAL_FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))["features"]
        self.assertNotIn("Vehicles", features)
        for stamp, result in self.results.items():
            end = pd.Timestamp(stamp)
            with self.subTest(stamp=stamp):
                self.assertEqual(len(result.junctions), 4)
                for state in result.junctions:
                    direct = predict_request(
                        {"junction": state.junction_id, "datetime": end.isoformat()}
                    )
                    self.assertAlmostEqual(
                        state.predicted_traffic,
                        float(direct["predicted_vehicles"]),
                        places=6,
                    )
                    self.assertEqual(
                        state.historical_traffic,
                        self.history[(state.junction_id, end)],
                    )
                    self.assertIsNotNone(state.incoming_influence)
                    self.assertNotAlmostEqual(
                        state.predicted_traffic,
                        float(state.incoming_influence),
                    )
                    print(
                        "STAGE7_JUNCTION"
                        f" {stamp} j={state.junction_id}"
                        f" predicted={state.predicted_traffic:.6f}"
                        f" historical={state.historical_traffic}"
                        f" influence={state.incoming_influence:.6f}"
                        f" sources={state.valid_source_count}"
                    )
                for item in result.relationships:
                    print(
                        "STAGE7_EDGE"
                        f" {stamp} {item.source_junction}->{item.target_junction}"
                        f" lag={item.lag_hours}"
                        f" r={item.correlation:.6f}"
                        f" abs={item.absolute_correlation:.6f}"
                        f" n={item.paired_observations}"
                    )

    def test_repeated_engine_runs_match(self) -> None:
        for stamp in REPEAT_STAMPS:
            with self.subTest(stamp=stamp):
                first = self.results[stamp]
                second = run_simulation(stamp)
                self.assertEqual(first.relationships, second.relationships)
                self.assertEqual(first.influence_result, second.influence_result)
                self.assertEqual(first.metadata, second.metadata)
                for left, right in zip(first.junctions, second.junctions):
                    self.assertEqual(left.historical_traffic, right.historical_traffic)
                    self.assertEqual(left.incoming_influence, right.incoming_influence)
                    self.assertAlmostEqual(left.predicted_traffic, right.predicted_traffic, places=6)

    def test_artifact_hashes_are_unchanged(self) -> None:
        for path, digest in self.hashes_before.items():
            self.assertEqual(_sha256(path), digest, path.name)


if __name__ == "__main__":
    unittest.main()
