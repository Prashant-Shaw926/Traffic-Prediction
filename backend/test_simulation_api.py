"""Flask tests for POST /api/simulation. Uses the real simulation engine."""

from __future__ import annotations

import unittest

from backend.app import app
from ml.simulation.config import default_config
from ml.simulation.data_provider import dataset_end_message, insufficient_history_message


class SimulationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = app.test_client()

    def _post(self, payload):
        return self.client.post("/api/simulation", json=payload)

    def _assert_success(self, stamp: str) -> dict:
        response = self._post({"datetime": stamp})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        body = response.get_json()
        self.assertEqual(len(body["junctions"]), 4)
        self.assertEqual(
            [item["junction_id"] for item in body["junctions"]],
            [1, 2, 3, 4],
        )
        self.assertGreaterEqual(len(body["relationships"]), 1)
        relationship = body["relationships"][0]
        self.assertIn("correlation", relationship)
        self.assertIn("absolute_correlation", relationship)
        self.assertIn("lag_hours", relationship)
        influence = body["influence_result"]
        self.assertEqual(influence["metadata"]["score_kind"], "influence_index")
        self.assertEqual(len(influence["junction_influences"]), 4)
        self.assertIn("contributions", influence["junction_influences"][0])
        self.assertIn("excluded_contributions", influence)
        metadata = body["metadata"]
        self.assertTrue(metadata["historical_simulation"])
        self.assertEqual(metadata["model"], "Spatial_GRU")
        self.assertEqual(metadata["features"], 53)
        self.assertEqual(metadata["lookback"], 168)
        self.assertEqual(metadata["influence_score_kind"], "influence_index")
        for junction in body["junctions"]:
            self.assertNotAlmostEqual(
                junction["predicted_traffic"],
                junction["incoming_influence"],
            )
            self.assertIsNotNone(junction["historical_traffic"])
        return body

    def test_june_15_simulation(self) -> None:
        self._assert_success("2017-06-15T10:00:00")

    def test_june_25_simulation(self) -> None:
        self._assert_success("2017-06-25T10:00:00")

    def test_repeated_request_matches(self) -> None:
        first = self._post({"datetime": "2017-06-15T10:00:00"}).get_json()
        second = self._post({"datetime": "2017-06-15T10:00:00"}).get_json()
        self.assertEqual(first, second)

    def test_missing_datetime_is_rejected(self) -> None:
        response = self._post({})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"],
            "Missing request fields: datetime is required.",
        )

    def test_malformed_datetime_is_rejected(self) -> None:
        response = self._post({"datetime": "not-a-datetime"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Invalid datetime.")

    def test_non_hourly_datetime_is_rejected(self) -> None:
        response = self._post({"datetime": "2017-06-15T10:30:00"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"],
            "Datetime must be aligned to a full hour.",
        )

    def test_insufficient_history_is_rejected(self) -> None:
        response = self._post({"datetime": "2017-01-01T00:00:00"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], insufficient_history_message())

    def test_timestamp_after_dataset_end_is_rejected(self) -> None:
        response = self._post({"datetime": "2017-07-01T10:00:00"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], dataset_end_message(default_config()))

    def test_existing_endpoints_still_succeed(self) -> None:
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json()["model"], "Spatial_GRU")

        locations = self.client.get("/api/locations")
        self.assertEqual(locations.status_code, 200)
        self.assertEqual(len(locations.get_json()["locations"]), 4)

        history = self.client.get("/api/history?junction=1")
        self.assertEqual(history.status_code, 200)
        self.assertGreater(len(history.get_json()["history"]), 0)

        prediction = self.client.post(
            "/api/predict",
            json={"junction": 1, "datetime": "2017-06-15T10:00:00"},
        )
        self.assertEqual(prediction.status_code, 200)
        body = prediction.get_json()
        self.assertEqual(body["model"], "Spatial_GRU")
        self.assertEqual(body["features"], 53)
        self.assertEqual(body["lookback"], 168)
        self.assertIn("predicted_vehicles", body)
        self.assertIn("points", body)


if __name__ == "__main__":
    unittest.main()
