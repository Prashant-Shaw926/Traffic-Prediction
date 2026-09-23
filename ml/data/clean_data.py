"""Clean the raw traffic CSV. Never overwrites data/raw/."""

from __future__ import annotations

from typing import Any

import pandas as pd

MAX_SHORT_GAP_HOURS = 3
FREQ = "h"


def parse_and_sort(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse DateTime, drop unparseable rows, sort by Junction then DateTime."""
    out = df.copy()
    original = len(out)
    out["DateTime"] = pd.to_datetime(out["DateTime"], errors="coerce")
    invalid = int(out["DateTime"].isna().sum())
    out = out.dropna(subset=["DateTime"])
    out["Junction"] = out["Junction"].astype(int)
    out = out.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    return out, {
        "original_rows": original,
        "invalid_timestamps_dropped": invalid,
    }


def drop_duplicate_keys(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep first row for duplicate (Junction, DateTime) keys."""
    n_dup = int(df.duplicated(subset=["Junction", "DateTime"]).sum())
    if n_dup:
        df = df.drop_duplicates(subset=["Junction", "DateTime"], keep="first")
    return df.reset_index(drop=True), n_dup


def _fill_short_gaps(series: pd.Series, max_hours: int) -> pd.Series:
    """Linear-interpolate runs of at most max_hours; leave longer runs as NaN."""
    missing = series.isna()
    if not missing.any():
        return series

    filled = series.copy()
    run_id = missing.ne(missing.shift(fill_value=False)).cumsum()
    for _, block in series[missing].groupby(run_id[missing]):
        if len(block) <= max_hours:
            filled.loc[block.index] = series.interpolate(method="time").loc[block.index]
    still = filled.isna()
    if still.any():
        run_id = still.ne(still.shift(fill_value=False)).cumsum()
        ffilled = filled.ffill()
        for _, block in filled[still].groupby(run_id[still]):
            if len(block) <= max_hours:
                filled.loc[block.index] = ffilled.loc[block.index]
    return filled


def regularize_hourly(
    df: pd.DataFrame,
    max_short_gap_hours: int = MAX_SHORT_GAP_HOURS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reindex each junction to hourly. Fill only gaps of <= max_short_gap_hours."""
    frames: list[pd.DataFrame] = []
    inserted = 0
    filled_short = 0
    dropped_long = 0
    per_junction: dict[str, Any] = {}

    for junction, group in df.groupby("Junction", sort=True):
        group = group.sort_values("DateTime").set_index("DateTime")
        full_index = pd.date_range(group.index.min(), group.index.max(), freq=FREQ)
        missing_before = int(len(full_index) - len(group))
        reindexed = group.reindex(full_index)
        reindexed.index.name = "DateTime"
        reindexed["Junction"] = junction
        vehicles = _fill_short_gaps(reindexed["Vehicles"], max_short_gap_hours)
        n_filled = int(reindexed["Vehicles"].isna().sum() - vehicles.isna().sum())
        reindexed["Vehicles"] = vehicles
        n_long = int(reindexed["Vehicles"].isna().sum())
        reindexed = reindexed.dropna(subset=["Vehicles"])
        reindexed = reindexed.reset_index()
        frames.append(reindexed)
        inserted += missing_before
        filled_short += n_filled
        dropped_long += n_long
        per_junction[str(junction)] = {
            "hourly_slots_inserted": missing_before,
            "short_gaps_filled": n_filled,
            "long_gap_rows_dropped": n_long,
        }

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    stats = {
        "hourly_slots_inserted": inserted,
        "short_gaps_filled": filled_short,
        "long_gap_rows_dropped": dropped_long,
        "max_short_gap_hours": max_short_gap_hours,
        "per_junction": per_junction,
        "rows_after_regularize": len(out),
    }
    return out, stats


def drop_nonphysical(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove only non-physical counts (Vehicles < 0). Peaks are kept."""
    bad = df["Vehicles"] < 0
    n = int(bad.sum())
    if n:
        df = df.loc[~bad]
    return df.reset_index(drop=True), n


def analyze_outliers(df: pd.DataFrame) -> dict[str, Any]:
    """IQR and Z-score flags per junction. Nothing is deleted here."""
    per_junction: dict[str, Any] = {}
    total_iqr = 0
    total_z = 0
    for junction, group in df.groupby("Junction", sort=True):
        vehicles = group["Vehicles"].astype(float)
        q1 = float(vehicles.quantile(0.25))
        q3 = float(vehicles.quantile(0.75))
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        iqr_mask = (vehicles < low) | (vehicles > high)
        mean = float(vehicles.mean())
        std = float(vehicles.std(ddof=0))
        z_mask = (vehicles - mean).abs() > 3 * std if std > 0 else vehicles != vehicles
        n_iqr = int(iqr_mask.sum())
        n_z = int(z_mask.sum())
        total_iqr += n_iqr
        total_z += n_z
        examples = group.loc[iqr_mask, ["DateTime", "Vehicles"]].head(5)
        per_junction[str(int(junction))] = {
            "n": int(len(group)),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "iqr_low": low,
            "iqr_high": high,
            "iqr_flagged": n_iqr,
            "zscore_flagged": n_z,
            "min": float(vehicles.min()),
            "max": float(vehicles.max()),
            "example_iqr_rows": [
                {"DateTime": str(row.DateTime), "Vehicles": float(row.Vehicles)}
                for row in examples.itertuples(index=False)
            ],
        }
    return {
        "method": "IQR 1.5 and |z|>3 per junction",
        "action": "retain all flagged rows; peak-hour spikes are valid traffic",
        "total_iqr_flagged": total_iqr,
        "total_zscore_flagged": total_z,
        "per_junction": per_junction,
    }


def clean_traffic(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full cleaning path. High vehicle counts are never dropped as outliers."""
    parsed, parse_stats = parse_and_sort(df)
    deduped, n_dup = drop_duplicate_keys(parsed)
    regularized, reg_stats = regularize_hourly(deduped)
    physical, n_neg = drop_nonphysical(regularized)
    outliers = analyze_outliers(physical)
    stats = {
        **parse_stats,
        "duplicate_junction_datetime_dropped": n_dup,
        **reg_stats,
        "nonphysical_negative_dropped": n_neg,
        "rows_after_clean": len(physical),
        "outliers": outliers,
    }
    return physical, stats


def format_outlier_report(stats: dict[str, Any]) -> str:
    outliers = stats["outliers"]
    lines = [
        "Traffic prediction - outlier report",
        "Method: IQR (1.5) and |z| > 3, computed per Junction.",
        "Action: all flagged rows RETAINED (except Vehicles < 0, which are non-physical).",
        "Peak-hour spikes are treated as genuine traffic, not errors.",
        "",
        f"IQR-flagged rows (retained): {outliers['total_iqr_flagged']}",
        f"Z-score-flagged rows (retained): {outliers['total_zscore_flagged']}",
        f"Non-physical Vehicles < 0 dropped: {stats['nonphysical_negative_dropped']}",
        "",
    ]
    for junction, info in outliers["per_junction"].items():
        lines.append(f"== Junction {junction} ==")
        lines.append(f"n={info['n']} min={info['min']:.0f} max={info['max']:.0f}")
        lines.append(
            f"Q1={info['q1']:.2f} Q3={info['q3']:.2f} IQR={info['iqr']:.2f} "
            f"fence=[{info['iqr_low']:.2f}, {info['iqr_high']:.2f}]"
        )
        lines.append(
            f"IQR flagged={info['iqr_flagged']}  Z-score flagged={info['zscore_flagged']}"
        )
        if info["example_iqr_rows"]:
            lines.append("Example IQR-flagged rows (kept):")
            for row in info["example_iqr_rows"]:
                lines.append(f"  {row['DateTime']}  Vehicles={row['Vehicles']:.0f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
