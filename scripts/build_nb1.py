"""
Generates all 8 Jupyter Notebooks across the required directories.
"""

import os
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(".").resolve()

def make_ks():
    return {
        "language_info": {"name": "python", "version": "3.13.7"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
    }

# ==============================================================================
# NOTEBOOK 1: 01_Data_Preprocessing/Data_Cleaning.ipynb
# ==============================================================================
def create_nb1():
    nb = nbf.v4.new_notebook(metadata=make_ks())
    cells = []
    
    cells.append(nbf.v4.new_markdown_cell("""# 01. Raw REFIT Data Inspection, Cleaning & 1-Minute Resampling

**Task:** Data Preprocessing & Quality Engineering (House 1)  
**Dataset:** Raw REFIT Electrical Load Measurements (University of Strathclyde)  
**Location:** `01_Data_Preprocessing/Data_Cleaning.ipynb`

---

## Executive Summary & Objectives
This notebook implements **Section 1 of the Flock Energy technical assignment**:
1. **Initial Inspection:** Load House 1 raw readings at ~8-second sampling; inspect dimensions, dtypes, and schema.
2. **Timestamp Validation:** Validate UTC datetime parsing, verify non-decreasing monotonicity, and detect retrograde steps.
3. **Data Quality Analysis:** Check duplicate timestamps, sampling interval distribution, missing readings, and long network outages.
4. **Outlier Detection:** Audit negative power values and unphysical extreme spikes (> 15 kW aggregate, > 4 kW sub-meters).
5. **Principled Cleaning Pipeline:**
   - Chronological sorting.
   - Deduplication rule: group identical timestamps and compute arithmetic mean power.
   - Clamp negative noise (< 0 W) to 0 W; cap unphysical spikes.
   - Resample to 1-minute intervals using bin-mean power.
   - Impute short gaps ($\le 3$ min) via linear time-interpolation; flag longer outages (> 3 min) as NaN with boolean `is_outage=True`.
6. **Before-and-After Visualisations:** Compare raw irregular telemetry vs. cleaned 1-minute signals for Aggregate, Appliance 5 (Washing Machine), and Appliance 6 (Dishwasher), saved in `outputs/cleaning/`."""))

    cells.append(nbf.v4.new_markdown_cell("""## 1. Environment Setup & Library Imports"""))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Plotting configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Relative paths
RAW_DIR = Path("../data/raw").resolve()
OUTPUTS_CLEANING = Path("../outputs/cleaning").resolve()
OUTPUTS_CLEANING.mkdir(parents=True, exist_ok=True)

print(f"Raw data directory: {RAW_DIR}")
print(f"Outputs cleaning directory: {OUTPUTS_CLEANING}")"""))

    cells.append(nbf.v4.new_markdown_cell("""## 2. Initial Data Loading & Inspection (House 1)"""))
    cells.append(nbf.v4.new_code_cell("""target_raw = RAW_DIR / "RAW_House1_Part1.csv"
assert target_raw.exists(), f"Raw file {target_raw} not found!"

# Column schema: Time, Unix, Aggregate, Appliance1..Appliance9
col_names = ["Time", "Unix", "Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]

print(f"Loading raw records from {target_raw.name}...")
df_raw = pd.read_csv(target_raw, nrows=350000, header=0, names=col_names)
print(f"Loaded DataFrame shape: {df_raw.shape}")
df_raw.head()"""))

    cells.append(nbf.v4.new_markdown_cell("""## 3. Timestamp Validation & Monotonicity Checks"""))
    cells.append(nbf.v4.new_code_cell("""df_raw['DateTime'] = pd.to_datetime(df_raw['Time'], errors='coerce')
invalid_dates = df_raw['DateTime'].isna().sum()

print("--- Timestamp Parsing Audit ---")
print(f"Total rows inspected:  {len(df_raw):,}")
print(f"Invalid / NaT count:   {invalid_dates}")
print(f"Start timestamp:       {df_raw['DateTime'].min()}")
print(f"End timestamp:         {df_raw['DateTime'].max()}")

# Monotonicity
is_monotonic = df_raw['DateTime'].is_monotonic_increasing
time_deltas = df_raw['DateTime'].diff().dt.total_seconds()
retrograde_steps = (time_deltas < 0).sum()
print(f"Strictly monotonic increasing: {is_monotonic}")
print(f"Retrograde steps (dt < 0):     {retrograde_steps}")"""))

    cells.append(nbf.v4.new_markdown_cell("""## 4. Duplicate Timestamp Detection & Handling Rule"""))
    cells.append(nbf.v4.new_code_cell("""dup_mask = df_raw['DateTime'].duplicated(keep=False)
total_dups = dup_mask.sum()
print(f"Rows with duplicate timestamps: {total_dups:,} ({total_dups / len(df_raw) * 100:.2f}%)")

# Example duplicate cluster
sample_dup_time = df_raw.loc[dup_mask, 'DateTime'].iloc[0]
print(f"\\nExample duplicate cluster at {sample_dup_time}:")
df_raw[df_raw['DateTime'] == sample_dup_time][['DateTime', 'Aggregate', 'Appliance5', 'Appliance6']]"""))

    cells.append(nbf.v4.new_markdown_cell("""### Stated Deduplication Rule:
> **Rule:** Group duplicate timestamps and compute the **arithmetic mean** across power channels.  
> **Rationale:** Preserves energy conservation without arbitrary selection bias from dropping readings."""))

    cells.append(nbf.v4.new_markdown_cell("""## 5. Sampling Interval Distribution & Outage Gap Analysis"""))
    cells.append(nbf.v4.new_code_cell("""df_dedup_sorted = df_raw.drop_duplicates('DateTime').sort_values('DateTime')
intervals = df_dedup_sorted['DateTime'].diff().dt.total_seconds().dropna()

print("--- Sampling Interval Statistics (Seconds) ---")
print(f"Median: {intervals.median():.1f} s | Mean: {intervals.mean():.2f} s | Min: {intervals.min():.1f} s | Max: {intervals.max():.1f} s")
print(f"<= 10s (Nominal):      {(intervals <= 10).sum():,} ({(intervals <= 10).mean()*100:.2f}%)")
print(f"10s - 60s (Jitter):    {((intervals > 10) & (intervals <= 60)).sum():,} ({((intervals > 10) & (intervals <= 60)).mean()*100:.2f}%)")
print(f"> 3 min (Outages):     {(intervals > 180).sum():,} ({(intervals > 180).mean()*100:.2f}%)")"""))

    cells.append(nbf.v4.new_markdown_cell("""## 6. Outlier & Impossible Value Detection"""))
    cells.append(nbf.v4.new_code_cell("""power_channels = ["Aggregate"] + [f"Appliance{i}" for i in range(1, 10)]
outlier_list = []

for ch in power_channels:
    vals = pd.to_numeric(df_raw[ch], errors='coerce')
    neg_count = (vals < 0).sum()
    limit = 15000.0 if ch == "Aggregate" else 4000.0
    spikes = (vals > limit).sum()
    outlier_list.append({
        "Channel": ch,
        "Min (W)": vals.min(),
        "Max (W)": vals.max(),
        "Negative (<0W)": neg_count,
        f"Spikes (>{int(limit)}W)": spikes
    })

df_outlier_summary = pd.DataFrame(outlier_list)
df_outlier_summary"""))

    cells.append(nbf.v4.new_markdown_cell("""## 7. Documented Data Cleaning & 1-Minute Resampling Pipeline"""))
    cells.append(nbf.v4.new_code_cell("""def clean_house1_pipeline(df_input, max_interp_mins=3):
    df = df_input.copy()
    df['DateTime'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['DateTime']).sort_values('DateTime')
    
    power_cols = ['Aggregate'] + [f'Appliance{i}' for i in range(1, 10)]
    for c in power_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # Deduplicate via mean
    df_dedup = df.groupby('DateTime')[power_cols].mean()
    
    # Bounding
    for c in power_cols:
        df_dedup.loc[df_dedup[c] < 0, c] = 0.0
        limit = 15000.0 if c == 'Aggregate' else 4000.0
        df_dedup.loc[df_dedup[c] > limit, c] = np.nan
        
    # 1-minute resampling
    df_1m = df_dedup.resample('1min').mean()
    
    # Gap interpolation <= 3 min; flag longer as outage
    df_clean = df_1m.interpolate(method='time', limit=max_interp_mins, limit_direction='forward')
    df_clean['is_outage'] = df_clean['Aggregate'].isna()
    
    # Rename channels for House 1
    rename_h1 = {
        'Appliance1': 'App_Fridge',
        'Appliance2': 'App_Freezer_1',
        'Appliance3': 'App_Freezer_2',
        'Appliance4': 'App_Washer_Dryer',
        'Appliance5': 'App_Washing_Machine',
        'Appliance6': 'App_Dishwasher',
        'Appliance7': 'App_Computer',
        'Appliance8': 'App_Television',
        'Appliance9': 'App_Electric_Heater'
    }
    df_clean = df_clean.rename(columns=rename_h1)
    return df_clean

df_h1_cleaned = clean_house1_pipeline(df_raw)
print(f"Cleaned 1-minute dataset shape: {df_h1_cleaned.shape}")
print(f"Outage bins: {df_h1_cleaned['is_outage'].sum():,} ({df_h1_cleaned['is_outage'].mean()*100:.2f}%)")
df_h1_cleaned.head()"""))

    cells.append(nbf.v4.new_markdown_cell("""## 8. Before-and-After Cleaning Visualisations"""))
    cells.append(nbf.v4.new_code_cell("""# 48-hour comparison window
sample_start = "2013-10-12 00:00:00"
sample_end   = "2013-10-14 00:00:00"

raw_slice = df_raw[(df_raw['DateTime'] >= sample_start) & (df_raw['DateTime'] <= sample_end)]
clean_slice = df_h1_cleaned.loc[sample_start:sample_end]

fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex='col', sharey='row')
fig.suptitle(
    f"House 1: Raw (~8s Irregular) vs. Cleaned & 1-Minute Resampled Signals\\nWindow: {sample_start} to {sample_end}",
    fontsize=14, fontweight='bold', y=0.98
)

c_agg, c_wm, c_dw = "#1f77b4", "#d62728", "#2ca02c"

# RAW PLOTS
axes[0, 0].plot(raw_slice['DateTime'], raw_slice['Aggregate'], color=c_agg, lw=0.7, alpha=0.85)
axes[0, 0].set_title("Raw: Aggregate Power (~8s)", fontsize=11, fontweight='semibold')
axes[0, 0].set_ylabel("Power (Watts)")
axes[0, 0].grid(True, linestyle="--", alpha=0.5)

axes[1, 0].plot(raw_slice['DateTime'], raw_slice['Appliance5'], color=c_wm, lw=0.8)
axes[1, 0].set_title("Raw: Washing Machine [Appliance 5] (~8s)", fontsize=11, fontweight='semibold')
axes[1, 0].set_ylabel("Power (Watts)")
axes[1, 0].grid(True, linestyle="--", alpha=0.5)

axes[2, 0].plot(raw_slice['DateTime'], raw_slice['Appliance6'], color=c_dw, lw=0.8)
axes[2, 0].set_title("Raw: Dishwasher [Appliance 6] (~8s)", fontsize=11, fontweight='semibold')
axes[2, 0].set_ylabel("Power (Watts)")
axes[2, 0].set_xlabel("Time (UTC)")
axes[2, 0].grid(True, linestyle="--", alpha=0.5)

# CLEANED PLOTS
axes[0, 1].plot(clean_slice.index, clean_slice['Aggregate'], color=c_agg, lw=1.2)
axes[0, 1].set_title("Cleaned & Resampled: Aggregate (1-min)", fontsize=11, fontweight='semibold')
axes[0, 1].grid(True, linestyle="--", alpha=0.5)

axes[1, 1].plot(clean_slice.index, clean_slice['App_Washing_Machine'], color=c_wm, lw=1.2)
axes[1, 1].set_title("Cleaned & Resampled: Washing Machine (1-min)", fontsize=11, fontweight='semibold')
axes[1, 1].grid(True, linestyle="--", alpha=0.5)

axes[2, 1].plot(clean_slice.index, clean_slice['App_Dishwasher'], color=c_dw, lw=1.2)
axes[2, 1].set_title("Cleaned & Resampled: Dishwasher (1-min)", fontsize=11, fontweight='semibold')
axes[2, 1].set_xlabel("Time (UTC)")
axes[2, 1].grid(True, linestyle="--", alpha=0.5)

for ax in axes.flat:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha='right')

plt.tight_layout()
fig_clean_path = OUTPUTS_CLEANING / "house1_before_after_cleaning.png"
fig.savefig(fig_clean_path, dpi=200, bbox_inches="tight")
print(f"Saved figure to {fig_clean_path}")
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""## 9. Findings & Quality Conclusion
1. **Timestamp Monotonicity:** Sorting resolves asynchronous arrival latency.
2. **Duplicate Handling:** Group-mean preserves area-under-the-curve energy integral.
3. **Outage Flagging:** Avoiding synthetic imputation on long gaps preserves valid ground truth for NILM modeling."""))

    nb.cells = cells
    out_file = Path("01_Data_Preprocessing/Data_Cleaning.ipynb")
    with open(out_file, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created {out_file}")

create_nb1()
