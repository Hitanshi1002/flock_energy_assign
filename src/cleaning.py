"""
Raw Data Inspection, Cleaning, and Resampling Pipeline for REFIT Dataset.

Implements Requirement 1:
- Timestamp parsing and monotonicity checks
- Duplicate timestamp detection and resolution
- Extreme and impossible value filtering
- 1-minute resampling with principled gap handling (short interpolation vs outage flagging)
- Before-and-after visualization
"""

import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import (
    HOUSE_APPLIANCES,
    MAX_VALID_AGGREGATE_POWER,
    MAX_VALID_IAM_POWER,
    MAX_INTERPOLATION_MINUTES,
    OUTAGE_THRESHOLD_MINUTES,
    CLEANING_FIG_DIR,
)

logger = logging.getLogger(__name__)


def load_raw_house_file(filepath: str) -> pd.DataFrame:
    """
    Loads a raw REFIT house file.
    Raw format: Time / DateTime, Unix timestamp, Aggregate, Appliance1..9
    Handles flexible header styles and whitespace delimiters.
    """
    # Raw REFIT CSVs may have headers or no headers, and either comma or space-delimited
    try:
        df = pd.read_csv(filepath, sep=None, engine='python', nrows=50)
        has_header = any("time" in str(col).lower() for col in df.columns)
    except Exception:
        has_header = False

    if has_header:
        df = pd.read_csv(filepath, sep=None, engine='python')
    else:
        df = pd.read_csv(
            filepath,
            sep=None,
            engine='python',
            names=["Time", "Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
        )

    # Standardize column names
    col_map = {}
    for col in df.columns:
        c_lower = str(col).strip().lower()
        if "time" in c_lower and "unix" not in c_lower:
            col_map[col] = "Time"
        elif "unix" in c_lower:
            col_map[col] = "Unix"
        elif "agg" in c_lower:
            col_map[col] = "Aggregate"
        elif "appliance" in c_lower:
            num = ''.join(filter(str.isdigit, c_lower))
            if num:
                col_map[col] = f"Appliance{num}"
    df = df.rename(columns=col_map)
    return df


def inspect_raw_data(df: pd.DataFrame, house_id: int = 1) -> Dict[str, Any]:
    """
    Performs comprehensive diagnostic inspection on raw data.
    Checks timestamps, monotonicity, duplicates, irregular intervals,
    missing spans, and impossible/extreme values.
    """
    report: Dict[str, Any] = {}
    report["house_id"] = house_id
    report["total_raw_rows"] = len(df)

    # 1. Timestamp Parsing & Ordering
    time_col = "Time" if "Time" in df.columns else df.columns[0]
    dt_series = pd.to_datetime(df[time_col], errors="coerce")
    invalid_dates = dt_series.isna().sum()
    report["invalid_timestamp_count"] = int(invalid_dates)

    valid_dt = dt_series.dropna()
    is_monotonic = valid_dt.is_monotonic_increasing
    report["is_strictly_monotonic_increasing"] = bool(is_monotonic)
    if not is_monotonic:
        time_diffs = valid_dt.diff().dt.total_seconds()
        retrograde_steps = (time_diffs < 0).sum()
        report["retrograde_timestamp_steps"] = int(retrograde_steps)
    else:
        report["retrograde_timestamp_steps"] = 0

    if len(valid_dt) > 0:
        report["start_time"] = str(valid_dt.min())
        report["end_time"] = str(valid_dt.max())
        report["timespan_days"] = round((valid_dt.max() - valid_dt.min()).total_seconds() / 86400, 2)

    # 2. Duplicate Timestamps
    dup_mask = dt_series.duplicated(keep=False)
    dup_count = dup_mask.sum()
    report["duplicate_timestamp_count"] = int(dup_count)
    report["duplicate_timestamp_pct"] = round(100.0 * dup_count / len(df), 4)

    # 3. Sampling Interval Distribution
    clean_dt = dt_series.drop_duplicates().sort_values()
    intervals = clean_dt.diff().dt.total_seconds().dropna()
    report["interval_median_sec"] = float(intervals.median()) if len(intervals) > 0 else 0.0
    report["interval_mean_sec"] = float(round(intervals.mean(), 2)) if len(intervals) > 0 else 0.0
    report["interval_min_sec"] = float(intervals.min()) if len(intervals) > 0 else 0.0
    report["interval_max_sec"] = float(intervals.max()) if len(intervals) > 0 else 0.0
    report["gaps_over_1min"] = int((intervals > 60).sum())
    report["gaps_over_15min"] = int((intervals > 900).sum())
    report["gaps_over_1hour"] = int((intervals > 3600).sum())

    # 4. Impossible and Extreme Values
    power_cols = [c for c in df.columns if c not in ["Time", "Unix", "Issues"]]
    report["power_columns"] = power_cols
    value_issues = {}

    for c in power_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        neg_count = (s < 0).sum()
        nan_count = s.isna().sum()
        max_val = float(s.max()) if not s.dropna().empty else 0.0
        min_val = float(s.min()) if not s.dropna().empty else 0.0

        limit = MAX_VALID_AGGREGATE_POWER if c == "Aggregate" else MAX_VALID_IAM_POWER
        extreme_spikes = (s > limit).sum()

        value_issues[c] = {
            "min_val": min_val,
            "max_val": max_val,
            "negative_readings": int(neg_count),
            "nan_readings": int(nan_count),
            "extreme_spikes_over_threshold": int(extreme_spikes),
            "threshold_watts": limit
        }
    report["value_issues"] = value_issues

    return report


def clean_raw_data_pipeline(
    df: pd.DataFrame,
    house_id: int = 1,
    resample_freq: str = "1min",
    max_interp_mins: int = MAX_INTERPOLATION_MINUTES,
    outage_threshold_mins: int = OUTAGE_THRESHOLD_MINUTES
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes a principled cleaning and resampling pipeline on raw house data.

    Steps:
    1. Parse datetime, drop unparseable rows, sort chronologically.
    2. Deduplicate timestamps using a clear stated rule:
       Rule: Group duplicate timestamps and calculate arithmetic mean of power channels.
       Rationale: Preserves energy conservation without arbitrary selection bias.
    3. Filter impossible and extreme values:
       - Clamp negative power values to 0.0 W (sensor calibration offsets).
       - Cap unrealistic spikes (> 15kW aggregate, > 4kW IAM) to threshold / NaN.
    4. Resample to continuous 1-minute grid using time-weighted or mean power.
    5. Handle missing gaps responsibly:
       - Short gaps (<= max_interp_mins): forward fill / linear interpolation.
       - Long outages (> max_interp_mins): marked as NaN, flagged with `is_outage=True`.
    6. Rename channels with real appliance names from metadata.

    Returns:
        (df_cleaned, cleaning_summary_dict)
    """
    summary: Dict[str, Any] = {}
    summary["raw_rows"] = len(df)

    # 1. Datetime Parsing & Sorting
    time_col = "Time" if "Time" in df.columns else df.columns[0]
    df = df.copy()
    df["Time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["Time"])
    df = df.sort_values("Time")
    summary["valid_datetime_rows"] = len(df)

    # Convert numeric power columns
    power_cols = [c for c in df.columns if c not in ["Time", "Unix", "Issues"]]
    for col in power_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Duplicate Resolution Rule
    dup_count_before = df["Time"].duplicated().sum()
    summary["duplicates_removed"] = int(dup_count_before)
    summary["duplicate_rule"] = "Group by exact timestamp and aggregate by mean power"
    df = df.groupby("Time", as_index=True)[power_cols].mean()

    # 3. Impossible and Extreme Value Handling
    summary["negative_values_clamped"] = 0
    summary["extreme_spikes_clipped"] = 0

    for col in power_cols:
        neg_mask = df[col] < 0
        summary["negative_values_clamped"] += int(neg_mask.sum())
        df.loc[neg_mask, col] = 0.0

        limit = MAX_VALID_AGGREGATE_POWER if col == "Aggregate" else MAX_VALID_IAM_POWER
        spike_mask = df[col] > limit
        summary["extreme_spikes_clipped"] += int(spike_mask.sum())
        df.loc[spike_mask, col] = np.nan

    # 4. Resample to 1-minute grid
    # Resample takes mean of observations falling into each 1-minute bin
    df_1min = df.resample(resample_freq).mean()
    summary["resampled_rows_total"] = len(df_1min)

    # 5. Gap and Outage Handling
    # Identify gaps: count consecutive NaNs
    # For short gaps (<= max_interp_mins), interpolate linearly
    # For longer gaps, mark as outage
    is_na_before = df_1min["Aggregate"].isna()
    summary["na_bins_before_interpolation"] = int(is_na_before.sum())

    # Limit linear interpolation to max_interp_mins consecutive minutes
    df_interp = df_1min.interpolate(method="time", limit=max_interp_mins, limit_direction="forward")

    # Flag remaining NaNs as outages
    is_outage = df_interp["Aggregate"].isna()
    df_interp["is_outage"] = is_outage
    summary["na_bins_after_interpolation"] = int(is_outage.sum())
    summary["outage_minutes"] = int(is_outage.sum())
    summary["outage_pct"] = round(100.0 * is_outage.sum() / len(df_interp), 2)
    summary["interpolation_rule"] = (
        f"Linear time-interpolation for short gaps up to {max_interp_mins} minutes. "
        f"Gaps exceeding {max_interp_mins} minutes are flagged as outages and kept as NaN."
    )

    # 6. Descriptive Renaming
    appliance_map = HOUSE_APPLIANCES.get(house_id, {})
    rename_dict = {}
    for i in range(1, 10):
        old_col = f"Appliance{i}"
        if old_col in df_interp.columns and i in appliance_map:
            rename_dict[old_col] = f"App_{appliance_map[i]}"
    df_interp = df_interp.rename(columns=rename_dict)

    return df_interp, summary


def plot_before_after_cleaning(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    house_id: int = 1,
    start_date: str = "2014-03-01",
    end_date: str = "2014-03-03",
    appliance_cols: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots aggregate power and at least two appliance channels before and after cleaning,
    satisfying Requirement 1.
    """
    # Prepare raw subset
    raw_time_col = "Time" if "Time" in df_raw.columns else df_raw.columns[0]
    df_raw_slice = df_raw.copy()
    df_raw_slice["dt"] = pd.to_datetime(df_raw_slice[raw_time_col], errors="coerce")
    df_raw_slice = df_raw_slice.dropna(subset=["dt"]).sort_values("dt")
    mask_raw = (df_raw_slice["dt"] >= start_date) & (df_raw_slice["dt"] <= end_date)
    raw_sub = df_raw_slice.loc[mask_raw]

    # Prepare cleaned subset
    mask_clean = (df_clean.index >= start_date) & (df_clean.index <= end_date)
    clean_sub = df_clean.loc[mask_clean]

    # Identify appliance columns to display
    if appliance_cols is None:
        if house_id == 1:
            appliance_cols = ["App_Washing_Machine", "App_Dishwasher"]
        else:
            cand = [c for c in df_clean.columns if c.startswith("App_")]
            appliance_cols = cand[:2] if len(cand) >= 2 else cand

    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col', sharey='row')
    fig.suptitle(
        f"House {house_id}: Raw vs. Cleaned & Resampled Power Comparison\n({start_date} to {end_date})",
        fontsize=14,
        fontweight="bold"
    )

    # Color palette
    c_agg = "#1f77b4"
    c_app1 = "#d62728"
    c_app2 = "#2ca02c"

    # RAW PLOTS (Left column)
    # 1. Aggregate
    axes[0, 0].plot(raw_sub["dt"], raw_sub["Aggregate"], color=c_agg, lw=0.8, alpha=0.85)
    axes[0, 0].set_title("Raw: Aggregate Power (Irregular ~8s)", fontsize=11, fontweight="semibold")
    axes[0, 0].set_ylabel("Power (Watts)")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Determine raw appliance channel names
    raw_app1 = "Appliance5" if "Appliance5" in raw_sub.columns else raw_sub.columns[3]
    raw_app2 = "Appliance6" if "Appliance6" in raw_sub.columns else raw_sub.columns[4]

    # 2. Appliance 1
    axes[1, 0].plot(raw_sub["dt"], pd.to_numeric(raw_sub[raw_app1], errors="coerce"), color=c_app1, lw=0.8)
    axes[1, 0].set_title(f"Raw: {raw_app1} (~8s)", fontsize=11, fontweight="semibold")
    axes[1, 0].set_ylabel("Power (Watts)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # 3. Appliance 2
    axes[2, 0].plot(raw_sub["dt"], pd.to_numeric(raw_sub[raw_app2], errors="coerce"), color=c_app2, lw=0.8)
    axes[2, 0].set_title(f"Raw: {raw_app2} (~8s)", fontsize=11, fontweight="semibold")
    axes[2, 0].set_ylabel("Power (Watts)")
    axes[2, 0].set_xlabel("Time (UTC)")
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)

    # CLEANED PLOTS (Right column)
    # 1. Aggregate
    axes[0, 1].plot(clean_sub.index, clean_sub["Aggregate"], color=c_agg, lw=1.2)
    # Highlight outages if any
    if "is_outage" in clean_sub.columns and clean_sub["is_outage"].any():
        outage_pts = clean_sub[clean_sub["is_outage"]]
        axes[0, 1].scatter(outage_pts.index, [0]*len(outage_pts), color="black", s=10, label="Outage / Gap > 3m")
        axes[0, 1].legend(loc="upper right")
    axes[0, 1].set_title("Cleaned & 1-Min Resampled: Aggregate Power", fontsize=11, fontweight="semibold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # 2. Appliance 1
    clean_app1_col = appliance_cols[0] if len(appliance_cols) > 0 and appliance_cols[0] in clean_sub.columns else clean_sub.columns[1]
    axes[1, 1].plot(clean_sub.index, clean_sub[clean_app1_col], color=c_app1, lw=1.2)
    axes[1, 1].set_title(f"Cleaned & 1-Min Resampled: {clean_app1_col}", fontsize=11, fontweight="semibold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    # 3. Appliance 2
    clean_app2_col = appliance_cols[1] if len(appliance_cols) > 1 and appliance_cols[1] in clean_sub.columns else clean_sub.columns[2]
    axes[2, 1].plot(clean_sub.index, clean_sub[clean_app2_col], color=c_app2, lw=1.2)
    axes[2, 1].set_title(f"Cleaned & 1-Min Resampled: {clean_app2_col}", fontsize=11, fontweight="semibold")
    axes[2, 1].set_xlabel("Time (UTC)")
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)

    # Formatting date axes
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Saved before-after cleaning figure to {save_path}")

    return fig
