"""Inspect the raw Kaggle traffic dataset. Does not clean or train."""

from __future__ import annotations

import pandas as pd

from ml.data.load_data import DatasetNotFoundError, RAW_CSV_PATH, REPO_ROOT, load_raw_traffic

REPORT_PATH = REPO_ROOT / "results" / "dataset_inspection.txt"

PAPER_FEATURES_NOT_IN_CSV = (
    "weather",
    "holiday / is_holiday",
    "road type",
    "GPS / latitude / longitude",
    "speed",
    "occupancy",
)


def _detect_datetime_column(df: pd.DataFrame) -> str | None:
    for name in df.columns:
        if name.lower().replace("_", "") in {"datetime", "timestamp", "date", "time"}:
            return name
    sample = df.iloc[: min(50, len(df))]
    for name in df.columns:
        if sample[name].dtype == object:
            converted = pd.to_datetime(sample[name], errors="coerce")
            if converted.notna().mean() >= 0.8:
                return name
    return None


def _detect_junction_column(df: pd.DataFrame) -> str | None:
    for name in df.columns:
        lowered = name.lower()
        if lowered in {"junction", "location", "location_id", "site"}:
            return name
    return None


def _detect_target_column(df: pd.DataFrame) -> str | None:
    preferred = {"vehicles", "vehicle_count", "volume", "traffic", "count"}
    numeric = df.select_dtypes(include="number").columns.tolist()
    for name in df.columns:
        if name.lower() in preferred and name in numeric:
            return name
    leftover = [c for c in numeric if c.lower() not in {"id", "junction"}]
    if len(leftover) == 1:
        return leftover[0]
    return leftover[0] if leftover else None


def _format_timedelta(delta: pd.Timedelta) -> str:
    seconds = delta.total_seconds()
    if pd.isna(seconds):
        return "unknown"
    if seconds % 3600 == 0:
        hours = int(seconds // 3600)
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    if seconds % 60 == 0:
        minutes = int(seconds // 60)
        return f"{minutes} minutes"
    return str(delta)


def _per_junction_frequency(df: pd.DataFrame, dt_col: str, junction_col: str | None) -> list[str]:
    lines: list[str] = []
    if junction_col is None:
        ordered = df.sort_values(dt_col)
        gaps = ordered[dt_col].diff().dropna()
        if gaps.empty:
            return ["Could not infer frequency (not enough timestamps)."]
        median = gaps.median()
        return [
            f"Overall median gap: {_format_timedelta(median)} "
            f"(min={_format_timedelta(gaps.min())}, max={_format_timedelta(gaps.max())})"
        ]

    for junction, group in df.groupby(junction_col, sort=True):
        ordered = group.sort_values(dt_col)
        gaps = ordered[dt_col].diff().dropna()
        if gaps.empty:
            lines.append(f"  Junction {junction}: n={len(group)}, frequency unknown")
            continue
        median = gaps.median()
        n_expected = None
        if median.total_seconds() > 0:
            span = ordered[dt_col].max() - ordered[dt_col].min()
            n_expected = int(span / median) + 1
        missing_slots = None if n_expected is None else n_expected - len(group)
        lines.append(
            f"  Junction {junction}: n={len(group)}, "
            f"{ordered[dt_col].min()} -> {ordered[dt_col].max()}, "
            f"median gap={_format_timedelta(median)}, "
            f"min gap={_format_timedelta(gaps.min())}, "
            f"max gap={_format_timedelta(gaps.max())}"
            + (f", estimated missing slots={missing_slots}" if missing_slots is not None else "")
        )
    return lines


def build_inspection_report(df: pd.DataFrame) -> str:
    dt_col = _detect_datetime_column(df)
    junction_col = _detect_junction_column(df)
    target_col = _detect_target_column(df)
    frequency_label = "unknown"
    parsed = df.copy()
    if dt_col is not None:
        parsed[dt_col] = pd.to_datetime(parsed[dt_col], errors="coerce")

    junction_col = _detect_junction_column(parsed)
    target_col = _detect_target_column(parsed)

    lines: list[str] = []
    lines.append("Traffic prediction - raw dataset inspection")
    lines.append(f"Source file: {RAW_CSV_PATH}")
    lines.append("")
    lines.append("== Shape ==")
    lines.append(f"rows={df.shape[0]}, columns={df.shape[1]}")
    lines.append("")
    lines.append("== Columns ==")
    lines.append(", ".join(df.columns.astype(str)))
    lines.append("")
    lines.append("== Dtypes ==")
    lines.append(df.dtypes.to_string())
    lines.append("")
    lines.append("== First 10 rows ==")
    lines.append(df.head(10).to_string(index=False))
    lines.append("")
    lines.append("== Last 10 rows ==")
    lines.append(df.tail(10).to_string(index=False))
    lines.append("")
    lines.append("== Missing values ==")
    missing = df.isna().sum()
    lines.append(missing.to_string())
    lines.append(f"total missing cells={int(missing.sum())}")
    lines.append("")
    lines.append("== Duplicate rows ==")
    lines.append(f"full-row duplicates={int(df.duplicated().sum())}")
    lines.append("")

    if junction_col is None:
        lines.append("== Junctions ==")
        lines.append("No junction/location column detected.")
        n_junctions = None
    else:
        unique = parsed[junction_col].dropna().unique()
        n_junctions = len(unique)
        lines.append("== Junctions ==")
        lines.append(f"column={junction_col}")
        lines.append(f"unique values ({n_junctions}): {sorted(unique.tolist(), key=str)}")
        counts = parsed[junction_col].value_counts().sort_index()
        lines.append(counts.to_string())
    lines.append("")

    if dt_col is None:
        lines.append("== Timestamps ==")
        lines.append("No timestamp column detected.")
        frequency_label = "unknown"
    else:
        invalid = parsed[dt_col].isna().sum()
        lines.append("== Timestamps ==")
        lines.append(f"column={dt_col}")
        lines.append(f"unparseable timestamps={int(invalid)}")
        valid = parsed.dropna(subset=[dt_col])
        lines.append(f"overall min={valid[dt_col].min()}")
        lines.append(f"overall max={valid[dt_col].max()}")
        lines.append("")
        lines.append("== Time frequency (computed, not assumed) ==")
        freq_lines = _per_junction_frequency(valid, dt_col, junction_col)
        lines.extend(freq_lines)
        # Overall median gap across junctions for a single label
        if junction_col is not None:
            medians = []
            for _, group in valid.groupby(junction_col):
                gaps = group.sort_values(dt_col)[dt_col].diff().dropna()
                if not gaps.empty:
                    medians.append(gaps.median())
            frequency_label = (
                _format_timedelta(pd.Series(medians).median()) if medians else "unknown"
            )
        else:
            gaps = valid.sort_values(dt_col)[dt_col].diff().dropna()
            frequency_label = _format_timedelta(gaps.median()) if not gaps.empty else "unknown"
        lines.append(f"inferred frequency (median of per-junction medians): {frequency_label}")
    lines.append("")

    lines.append("== Target candidate ==")
    if target_col is None:
        lines.append("No numeric vehicle-count column detected.")
    else:
        lines.append(f"column={target_col}")
        series = pd.to_numeric(parsed[target_col], errors="coerce")
        lines.append(series.describe().to_string())
    lines.append("")

    lines.append("== Paper features unavailable in this CSV ==")
    for item in PAPER_FEATURES_NOT_IN_CSV:
        present = any(item.split()[0].lower() in c.lower() for c in df.columns)
        if not present:
            lines.append(f"- {item}: not present (do not fabricate)")
    lines.append("")

    lines.append("== Proposed feature set (next slice, not implemented here) ==")
    proposed: list[str] = []
    if dt_col is not None:
        proposed.extend(
            [
                "hour, day, month, day_of_week, is_weekend (from timestamp)",
                "hour_sin, hour_cos (cyclic encoding of hour)",
            ]
        )
    if frequency_label.endswith("hour") or frequency_label.endswith("hours"):
        proposed.append(
            "lags at t-1, t-2, t-3 (previous hours), t-24 (previous day), "
            "t-168 (previous week) - only because frequency is hourly"
        )
    elif frequency_label != "unknown":
        proposed.append(
            f"lags must be expressed in rows matching {frequency_label}; "
            "do not treat t-24 as 'yesterday' unless frequency is hourly"
        )
    if junction_col is not None:
        proposed.append(f"{junction_col} as a location identifier")
    if target_col is not None:
        proposed.append(f"target: {target_col} (traffic volume / vehicle count)")
    for item in proposed:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("== Proposed preprocessing pipeline (next slice, not implemented here) ==")
    lines.append("1. Parse timestamps; drop or flag unparseable rows")
    lines.append("2. Deduplicate full rows; resolve duplicate timestamp+junction if any")
    lines.append("3. Per-junction regularize to the inferred frequency; interpolate/forward-fill short gaps")
    lines.append("4. Outlier review with IQR/Z-score, but do not drop valid peak-hour spikes")
    lines.append("5. Engineer only the proposed features above")
    lines.append("6. Temporal split 70/15/15 (train -> val -> test in time); no shuffle")
    lines.append("7. Fit MinMaxScaler on train only; transform val/test; persist scaler")
    lines.append("8. Save to data/processed/traffic_processed.csv; never overwrite data/raw/")
    lines.append("")
    lines.append("Inspection only. No cleaning, training, or metrics were produced.")
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        df = load_raw_traffic()
    except DatasetNotFoundError as exc:
        print(str(exc))
        return 1

    report = build_inspection_report(df)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
