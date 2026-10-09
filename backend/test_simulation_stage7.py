"""Stage 7 Flask checks. Compares POST /api/simulation with the Python engine."""

from __future__ import annotations

import hashlib
import math
import unittest
from pathlib import Path

import pandas as pd

from backend.app import app
from backend.config import (
    FEATURE_SCALER_PATH,
    GRU_MODEL_PATH,
    SPATIAL_FEATURE_CONFIG_PATH,
    SPATIAL_FEATURE_SCALER_PATH,
    SPATIAL_GRU_MODEL_PATH,
    SPATIAL_TARGET_SCALER_PATH,
    TARGET_SCALER_PATH,
)
from backend.services.simulation_api import simulation_response
from ml.simulation.config import default_config
from ml.simulation.data_provider import dataset_end_message, insufficient_history_message
from ml.simulation.simulation_engine import run_simulation

REPEAT_STAMPS = ("2017-06-15T10:00:00", "2017-06-25T10:00:00")
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


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


class Stage7ApiConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = app.test_client()
        cls.hashes_before = {path: _sha256(path) for path in ARTIFACTS}

    def _post(self, payload):
        return self.client.post("/api/simulation", json=payload)

    def test_repeated_posts_match_the_engine(self) -> None:
        for stamp in REPEAT_STAMPS:
            with self.subTest(stamp=stamp):
                first = self._post({"datetime": stamp})
                second = self._post({"datetime": stamp})
                self.assertEqual(first.status_code, 200, first.get_data(as_text=True))
                self.assertEqual(first.get_json(), second.get_json())
                direct = simulation_response({"datetime": stamp})
                self.assertEqual(first.get_json(), direct)
                engine = run_simulation(stamp)
                body = first.get_json()
                self.assertEqual(body["timestamp"], pd.Timestamp(stamp).isoformat())
                self.assertEqual(len(body["junctions"]), len(engine.junctions))
                self.assertEqual(len(body["relationships"]), len(engine.relationships))
                for item, state in zip(body["junctions"], engine.junctions):
                    self.assertEqual(item["junction_id"], state.junction_id)
                    self.assertTrue(_close(item["predicted_traffic"], state.predicted_traffic))
                    self.assertEqual(item["historical_traffic"], state.historical_traffic)
                    self.assertTrue(_close(item["incoming_influence"], state.incoming_influence))
                    self.assertEqual(item["valid_source_count"], state.valid_source_count)
                for item, relationship in zip(body["relationships"], engine.relationships):
                    self.assertEqual(item["source_junction"], relationship.source_junction)
                    self.assertEqual(item["target_junction"], relationship.target_junction)
                    self.assertEqual(item["lag_hours"], relationship.lag_hours)
                    self.assertTrue(_close(item["correlation"], relationship.correlation))
                    self.assertTrue(
                        _close(item["absolute_correlation"], relationship.absolute_correlation)
                    )
                    self.assertEqual(item["paired_observations"], relationship.paired_observations)
                influences = body["influence_result"]["junction_influences"]
                for item, state in zip(influences, engine.influence_result.junction_influences):
                    self.assertTrue(_close(item["incoming_influence"], state.incoming_influence))
                    self.assertEqual(len(item["contributions"]), len(state.contributions))
                    for raw, contribution in zip(item["contributions"], state.contributions):
                        self.assertEqual(raw["lag_hours"], contribution.lag_hours)
                        self.assertTrue(_close(raw["contribution"], contribution.contribution))
                        self.assertTrue(_close(raw["source_traffic"], contribution.source_traffic))
                        self.assertTrue(
                            _close(raw["signed_correlation"], contribution.signed_correlation)
                        )

    def test_boundary_errors_keep_backend_text(self) -> None:
        cases = (
            ("not-a-datetime", "Invalid datetime."),
            ("", "Missing request fields: datetime is required."),
            ("2017-06-15T10:30:00", "Datetime must be aligned to a full hour."),
            ("2017-01-01T00:00:00", insufficient_history_message()),
            ("2017-07-01T10:00:00", dataset_end_message(default_config())),
            ("2026-10-09T10:00:00", dataset_end_message(default_config())),
        )
        for value, message in cases:
            with self.subTest(value=value):
                response = self._post({"datetime": value})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"], message)

    def test_predict_endpoint_still_succeeds(self) -> None:
        response = self.client.post(
            "/api/predict",
            json={"junction": 1, "datetime": "2017-06-15T10:00:00"},
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        body = response.get_json()
        self.assertEqual(body["model"], "Spatial_GRU")
        self.assertEqual(body["features"], 53)
        self.assertEqual(body["lookback"], 168)
        self.assertIn("predicted_vehicles", body)

    def test_artifact_hashes_are_unchanged(self) -> None:
        for path, digest in self.hashes_before.items():
            self.assertEqual(_sha256(path), digest, path.name)
            print(f"STAGE7_HASH {path.name} {digest}")


if __name__ == "__main__":
    unittest.main()
