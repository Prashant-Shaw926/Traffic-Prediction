"""Load real historical traffic and build a leakage-safe simulation window.

Stage 1 does not predict and does not calculate junction relationships.
Every vehicle count is copied from data/raw/traffic.csv.
"""

from __future__ import annotations

import pandas as pd

from ml.data.load_data import load_raw_traffic
from ml.simulation.config import SimulationConfig, default_config
from ml.simulation.models import (
    JunctionTrafficState,
    SimulationData,
    SimulationRequest,
    SimulationWindow,
    TrafficObservation,
)

REQUIRED_COLUMNS = ("DateTime", "Junction", "Vehicles")


class SimulationInputError(ValueError):
    """The requested simulation timestamp cannot be turned into a valid window."""


def dataset_end_message(config: SimulationConfig) -> str:
    label = pd.Timestamp(config.dataset_end).strftime("%Y-%m-%d %H:%M")
    return (
        "The historical dataset ends at "
        f"{label} and does not contain sufficient data for this timestamp."
    )


def insufficient_history_message() -> str:
    return "Insufficient historical data for the requested simulation timestamp."


def parse_simulation_timestamp(value: object) -> pd.Timestamp:
    """Require a real, hour-aligned timestamp. Does not check the dataset range."""
    if value is None or str(value).strip() == "":
        raise SimulationInputError("Invalid datetime.")
    try:
        stamp = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError) as exc:
        raise SimulationInputError("Invalid datetime.") from exc
    if not isinstance(stamp, pd.Timestamp) or pd.isna(stamp):
        raise SimulationInputError("Invalid datetime.")
    if stamp.tzinfo is not None:
        stamp = pd.Timestamp(stamp.to_pydatetime().replace(tzinfo=None))
    stamp = pd.Timestamp(stamp).floor("s")
    if stamp.minute != 0 or stamp.second != 0 or stamp.microsecond != 0:
        raise SimulationInputError("Datetime must be aligned to a full hour.")
    return stamp


def window_bounds(timestamp: pd.Timestamp, config: SimulationConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    if config.historical_window_hours < 1:
        raise SimulationInputError("historical_window_hours must be at least 1.")
    start = timestamp - pd.Timedelta(hours=config.historical_window_hours - 1)
    return start, timestamp


def load_historical_traffic(config: SimulationConfig | None = None) -> pd.DataFrame:
    """Return cleaned-but-not-imputed rows sorted by junction, then time."""
    settings = config if config is not None else default_config()
    frame = load_raw_traffic()
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SimulationInputError(
            "Historical traffic is missing required columns: " + ", ".join(missing)
        )

    out = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
    out["Vehicles"] = pd.to_numeric(out["Vehicles"], errors="coerce")
    out["Junction"] = pd.to_numeric(out["Junction"], errors="coerce")
    out = out.dropna(subset=["DateTime", "Junction", "Vehicles"])
    out["Junction"] = out["Junction"].astype(int)
    allowed = set(settings.supported_junctions)
    out = out.loc[out["Junction"].isin(allowed)]
    out = out.sort_values(["Junction", "DateTime"], kind="mergesort")
    out = out.drop_duplicates(subset=["Junction", "DateTime"], keep="first")
    return out.reset_index(drop=True)


def build_simulation_input(
    timestamp: object,
    config: SimulationConfig | None = None,
) -> SimulationData:
    """Build the historical state visible at `timestamp`.

    Rows after the timestamp are excluded before the window is sliced.
    Missing hours are left missing; they are not interpolated.
    """
    settings = config if config is not None else default_config()
    if settings.minimum_observations < 1:
        raise SimulationInputError("minimum_observations must be at least 1.")
    if not settings.supported_junctions:
        raise SimulationInputError("supported_junctions must not be empty.")

    stamp = parse_simulation_timestamp(timestamp)
    configured_end = pd.Timestamp(settings.dataset_end)
    history = load_historical_traffic(settings)
    if history.empty:
        raise SimulationInputError(dataset_end_message(settings))

    latest = pd.Timestamp(history["DateTime"].max())
    if stamp > configured_end or stamp > latest:
        raise SimulationInputError(dataset_end_message(settings))

    start, end = window_bounds(stamp, settings)
    states: list[JunctionTrafficState] = []
    for junction in settings.supported_junctions:
        series = history.loc[history["Junction"] == junction, ["DateTime", "Vehicles"]]
        series = series.sort_values("DateTime", kind="mergesort")
        available = series.loc[series["DateTime"] <= stamp]
        if not available.empty and pd.Timestamp(available["DateTime"].max()) > stamp:
            raise AssertionError("Future rows remained after the history cutoff")
        window_rows = available.loc[available["DateTime"] >= start]
        if (window_rows["DateTime"] > stamp).any():
            raise AssertionError("Future rows entered the simulation window")
        has_endpoint = (window_rows["DateTime"] == stamp).any()
        if (not has_endpoint) or len(window_rows) < settings.minimum_observations:
            raise SimulationInputError(insufficient_history_message())
        observations = tuple(
            TrafficObservation(
                timestamp=pd.Timestamp(row.DateTime),
                vehicles=float(row.Vehicles),
            )
            for row in window_rows.itertuples(index=False)
        )
        states.append(JunctionTrafficState(junction=int(junction), observations=observations))

    return SimulationData(
        request=SimulationRequest(timestamp=stamp),
        window=SimulationWindow(start=start, end=end, states=tuple(states)),
        config=settings,
    )
