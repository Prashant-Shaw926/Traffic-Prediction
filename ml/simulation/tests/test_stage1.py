"""Unit tests for the Stage 1 simulation input window. Uses the real CSV."""

from __future__ import annotations

import unittest

import pandas as pd

from ml.simulation import SimulationInputError, build_simulation_input
from ml.simulation.data_provider import (
    dataset_end_message,
    insufficient_history_message,
    load_historical_traffic,
)


VALID = "2017-06-15 10:00:00"


class Stage1SimulationInputTests(unittest.TestCase):
    def test_valid_timestamp_has_four_junctions_in_order_without_future_rows(self) -> None:
        data = build_simulation_input(VALID)
        end = pd.Timestamp(VALID)
        self.assertEqual(data.request.timestamp, end)
        self.assertEqual(data.window.end, end)
        self.assertEqual(data.window.start, end - pd.Timedelta(hours=167))
        self.assertEqual(
            tuple(state.junction for state in data.window.states),
            (1, 2, 3, 4),
        )

        history = load_historical_traffic(data.config)
        for state in data.window.states:
            timestamps = [item.timestamp for item in state.observations]
            self.assertGreaterEqual(len(timestamps), data.config.minimum_observations)
            self.assertEqual(timestamps, sorted(timestamps))
            self.assertTrue(all(item <= end for item in timestamps))
            self.assertEqual(timestamps[-1], end)
            self.assertLess(timestamps[0], end)
            expected = history.loc[
                (history["Junction"] == state.junction)
                & (history["DateTime"] >= data.window.start)
                & (history["DateTime"] <= end)
            ]
            self.assertEqual(
                [item.vehicles for item in state.observations],
                [float(value) for value in expected["Vehicles"].tolist()],
            )

    def test_timestamp_after_dataset_end_is_rejected(self) -> None:
        message = dataset_end_message(build_simulation_input(VALID).config)
        self.assertIn("2017-06-30 23:00", message)
        for value in ("2017-07-01 10:00:00", "2026-10-02T10:00:00"):
            with self.subTest(value=value):
                with self.assertRaises(SimulationInputError) as caught:
                    build_simulation_input(value)
                self.assertEqual(str(caught.exception), message)

    def test_timestamp_before_sufficient_history_is_rejected(self) -> None:
        with self.assertRaises(SimulationInputError) as caught:
            build_simulation_input("2017-01-01 00:00:00")
        self.assertEqual(str(caught.exception), insufficient_history_message())

    def test_malformed_timestamp_is_rejected(self) -> None:
        with self.assertRaises(SimulationInputError) as caught:
            build_simulation_input("not-a-datetime")
        self.assertEqual(str(caught.exception), "Invalid datetime.")

    def test_non_hourly_timestamp_is_rejected(self) -> None:
        with self.assertRaises(SimulationInputError) as caught:
            build_simulation_input("2017-06-15 10:30:00")
        self.assertEqual(str(caught.exception), "Datetime must be aligned to a full hour.")

    def test_same_timestamp_is_deterministic(self) -> None:
        first = build_simulation_input(VALID)
        second = build_simulation_input(VALID)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
