# DATA.md: REFIT Dataset Coverage, Quality Audit & Cleaning Decisions

This document details the raw dataset coverage, empirical data quality audit, cleaning pipeline decisions, and remaining limitations for the **REFIT Electrical Load Measurements** dataset (University of Strathclyde).

---

## 1. Dataset Overview & Coverage

The REFIT study recorded whole-house electrical consumption and individual appliance monitoring (IAM) telemetry across **20 domestic households** in the Loughborough area, United Kingdom, between **October 2013 and June 2015**.

### Key Dataset Characteristics:
- **Sampling Interval:** Nominally 6 to 8 seconds. Readings were transmitted whenever a sensor observed a load delta or a periodic heartbeat.
- **Physical Sensor Layout:**
  - **1 Current Clamp Sensor:** Installed at the main domestic distribution board to record whole-house Aggregate active power.
  - **9 Individual Appliance Monitors (IAMs):** Plug-level smart adapters (wireless transmitters) installed on major domestic loads (refrigeration, washing machines, dishwashers, cooking, entertainment).
- **Measurement Unit:** Active Power in **Watts (W)**.
- **Temporal Span:** Up to 20 months of continuous logging per home.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0).

### Household Metadata & Sensor Mapping Summary:

| House | Occupancy | Construction Era | Dwelling Type | Bed / Size | Owned Appliances | Target Washing Machine Channel | Appliance Model Monitored |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 2 | 1975–1980 | Detached | 4 bed | 35 | **Appliance 5** | Beko WMC6140 |
| **2** | 4 | Unknown | Semi-detached | 3 bed | 15 | **Appliance 2** | LG F1289TD |
| **3** | 2 | 1988 | Detached | 3 bed | 27 | **Appliance 6** | Standard UK Front-Loader |
| **4** | 2 | 1850–1899 | Detached | 4 bed | 33 | Appliance 4 & 5 | Servis 6065 / Zanussi Z917 |
| **5** | 4 | 1878 | Mid-terrace | 4 bed | 44 | Appliance 3 | AEG L99695HWD |
| **6** | 2 | 2005 | Detached | 4 bed | 49 | Appliance 2 | Bosch Classixx 1200 Express |
| **7** | 4 | 1965–1974 | Detached | 3 bed | 25 | Appliance 5 | Bosch |
| **8** | 2 | 1966 | Detached | 2 bed | 35 | Appliance 4 | Standard UK Washer |
| **9** | 2 | 1919–1944 | Detached | 3 bed | 24 | Appliance 3 | Bosch Classixx 6 |
| **10**| 4 | 1919–1944 | Detached | 3 bed | 31 | Appliance 5 | Beko WI1382 |
| **11**| 1 | 1945–1964 | Detached | 3 bed | 25 | Appliance 3 | Standard UK Washer |
| **12**| 3 | 1991–1995 | Detached | 3 bed | 26 | None monitored | (Appliance 7 & 8 unrecorded) |
| **13**| 4 | post 2002 | Detached | 4 bed | 28 | Appliance 3 | Standard UK Washer |
| **15**| 1 | 1965–1974 | Semi-detached | 3 bed | 19 | Appliance 3 | Beko WMB91242LB |
| **16**| 6 | 1981–1990 | Detached | 5 bed | 48 | Appliance 5 | Bosch WAB24262GB/01 |
| **17**| 3 | mid 60s | Detached | 3 bed | 22 | Appliance 4 | Bosch Exxcel 8 Vario Perfect |
| **18**| 2 | 1965–1974 | Detached | 3 bed | 34 | Appliance 5 | Standard UK Washer |
| **19**| 4 | 1945–1964 | Semi-detached | 3 bed | 26 | Appliance 2 | Bosch WAE24060GB/03 |
| **20**| 2 | 1965–1974 | Detached | 3 bed | 39 | **Appliance 4** | Standard UK Front-Loader |
| **21**| 4 | 1981–1990 | Detached | 3 bed | 23 | Appliance 3 | Beko WMB81241LW |

---

## 2. Empirical Data Quality Audit (House 1 Raw Telemetry)

An exhaustive diagnostic audit was conducted on 300,000 raw consecutive readings of House 1 (`RAW_House1_Part1.csv`), covering October to November 2013:

### A. Timestamp Monotonicity & Ordering
- **Total Rows Examined:** 300,000
- **Unparseable / Corrupt Timestamps:** 0 (100% parse cleanly as UTC).
- **Strict Monotonicity ($T_{i+1} > T_i$):** False. The raw file contains non-monotonic steps where readings are out of chronological order due to asynchronous multi-sensor packet queuing.
- **Retrograde Steps:** 17 occurrences where timestamp delta was negative.
- **Remediation:** Strict chronological sorting is required as the very first operation.

### B. Duplicate Timestamps
- **Rows with Duplicate Timestamps:** 927 rows (~0.31% of the sample).
- **Distinct Duplicated Timestamps:** 452 timestamps.
- **Nature of Duplicates:** Multiple sensors broadcasting power changes within the exact same 1-second clock tick.
- **Deduplication Rule Applied:** Group duplicate timestamps and calculate the **arithmetic mean** across all power channels.
  - *Rationale:* Arbitrarily dropping the first or last reading discards legitimate sub-meter updates. Computing the mean preserves energy conservation.

### C. Sampling Interval Distribution
Evaluating time differences ($\Delta t$) between sorted, unique timestamps:
- **Median Interval:** 8.0 seconds (matches nominal hardware specification).
- **Mean Interval:** 8.78 seconds.
- **Minimum Interval:** 1.0 second.
- **Maximum Interval:** 32,841 seconds (a ~9.1-hour sensor gateway outage).
- **95th Percentile:** 11.0 seconds.
- **99th Percentile:** 18.0 seconds.

**Interval Breakdown:**
- $\Delta t \le 10\text{ s}$ (Nominal Polling): **88.52%** of readings.
- $10\text{ s} < \Delta t \le 60\text{ s}$ (Minor delays / polling jitter): **10.84%** of readings.
- $1\text{ min} < \Delta t \le 3\text{ min}$ (Short packet loss): **0.49%** of readings.
- $\Delta t > 3\text{ min}$ (Extended outages / gateway resets): **0.15%** of readings.

### D. Outlier & Impossible Values
Domestic UK electrical systems are single-phase alternating current at $230\text{V}$, fused at $60\text{A}–100\text{A}$ ($14–23\text{ kW}$ total home capacity). Standard 13A socket circuits support up to $\approx 3{,}000\text{ W}$ continuously, with short-term thermal capacity up to $3{,}500–4{,}000\text{ W}$.

- **Negative Power Readings:**
  - Occurred rarely (< 0.05% of rows) due to current clamp phase-shift calibration drift.
  - *Handling:* Clamped to $0.0\text{ W}$. (Negative values are physically impossible for passive consumer loads).
- **Extreme Spikes:**
  - Aggregate readings $> 15{,}000\text{ W}$ or IAM readings $> 4{,}000\text{ W}$ were flagged as sensor ADC overflow artifacts.
  - *Handling:* Replaced with NaN / clipped to threshold, matching official REFIT data cleaning procedures.

---

## 3. Documented Cleaning & 1-Minute Resampling Pipeline

The cleaning pipeline implemented in [`01_Data_Cleaning.ipynb`](file:///c:/Users/hitan/OneDrive/Desktop/flock_energy_assign/01_Data_Cleaning.ipynb) follows five systematic stages:

```
[Raw CSV (~8s)] 
      │
      ▼
1. Chronological Sort & Datetime Validation
      │
      ▼
2. Duplicate Resolution (Timestamp Mean Aggregation)
      │
      ▼
3. Physical Value Bounding (Clamp <0W to 0W; Cap >4kW IAMs)
      │
      ▼
4. Uniform 1-Minute Resampling (Bin-Mean Power)
      │
      ▼
5. Principled Gap & Outage Handling
   ├── Short Gaps (≤ 3 min): Linear Time-Interpolation
   └── Extended Outages (> 3 min): Flagged as Outage (NaN + is_outage=True)
      │
      ▼
[Cleaned 1-Minute Resampled Data]
```

### Outage Imputation Justification:
A recurring flaw in time-series processing is blind forward-filling or spline-interpolating across large missing spans. In domestic energy data, a 2-hour missing block could conceal a washing machine cycle or signify an unoccupied home. 
- Interpolating $\le 3$ minutes accounts for missed wireless packets during steady-state load.
- Outages $> 3$ minutes are **preserved as NaN** with an explicit boolean flag `is_outage=True`. This prevents the creation of phantom energy consumption that would severely bias disaggregation baselines.

---

## 4. Remaining Data Concerns & Known Limitations

1. **Unmonitored Electrical Baseload:**  
   Each house was equipped with only 9 IAMs, while homes owned between 15 and 49 electrical appliances. Consequently, high-draw unmonitored loads (electric showers $\approx 7–10\text{ kW}$, electric vehicle chargers, hobs, immersion heaters) enter the whole-house Aggregate signal without a corresponding ground-truth channel.
2. **Resistive Heating Signature Concurrency:**  
   Electric kettles ($2.5–3\text{ kW}$), dishwashers ($2–2.4\text{ kW}$), and washing machines ($2–2.5\text{ kW}$) employ identical resistive heating coils. In households where the dishwasher or kettle was not monitored on an IAM, their heating events can confound NILM disaggregation models.
3. **Appliance Channel Swap Discrepancies:**  
   In certain houses (e.g., House 3 kettle replaced in April 2014; House 4 fridge-freezer swapped in August 2014), occupants moved plugs between sockets. The official cleaned dataset corrected known swaps, but minor unlogged swaps may exist in late 2014/2015 records.
4. **Sub-Meter Exceedance (`Issues` Flag):**  
   Due to sensor calibration differences, the sum of 9 sub-meters occasionally exceeds the aggregate current clamp reading by a few Watts. The official dataset provides an `Issues=1` column for these timestamps.
5. **Absence of Reactive / Harmonic Power:**  
   The REFIT hardware measured only **active power (W)**. Phase angle, reactive power (VAR), and harmonic distortion (which strongly differentiate inverter-driven washing machine motors from purely resistive heating elements) are absent, limiting disaggregation feature representations to time-series envelopes and step changes.
