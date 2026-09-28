"""Historical traffic from processed CSVs. No database and no fabricated rows."""

from __future__ import annotations

import pandas as pd

from backend.config import HISTORY_DEFAULT_HOURS, LOCATION_BY_JUNCTION
from backend.services.prediction_service import (
    artifacts,
    location_id_for_junction,
    parse_junction,
    parse_timestamp,
)
from backend.utils.responses import ApiError, congestion_from_volume, isoformat


def history_request(
    *,
    junction=None,
    location_id=None,
    start=None,
    end=None,
) -> dict:
    junction_id = parse_junction(junction=junction, location_id=location_id)
    frame = artifacts().frame
    series = frame.loc[frame["Junction"].astype(int) == junction_id].sort_values(
        "DateTime"
    )
    if series.empty:
        raise ApiError("Invalid junction.", 400)

    if (start is None or str(start).strip() == "") and (
        end is None or str(end).strip() == ""
    ):
        last = pd.to_datetime(series["DateTime"]).max()
        start_ts = last - pd.Timedelta(hours=HISTORY_DEFAULT_HOURS - 1)
        end_ts = last
    else:
        start_ts = parse_timestamp(start, field="start")
        end_ts = parse_timestamp(end if end not in (None, "") else start, field="end")
        if end_ts < start_ts:
            raise ApiError("End must be on or after start.", 400)

    times = pd.to_datetime(series["DateTime"])
    selected = series.loc[(times >= start_ts) & (times <= end_ts)]
    history = []
    for row in selected.itertuples(index=False):
        vehicles = float(row.Vehicles)
        stamp = isoformat(row.DateTime)
        history.append(
            {
                "datetime": stamp,
                "timestamp": stamp,
                "vehicles": vehicles,
                "actual": vehicles,
                "congestion": congestion_from_volume(vehicles),
            }
        )
    return {
        "junction": junction_id,
        "locationId": location_id_for_junction(junction_id),
        "name": LOCATION_BY_JUNCTION[junction_id]["name"],
        "start": isoformat(start_ts),
        "end": isoformat(end_ts),
        "history": history,
    }
