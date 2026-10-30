# Flock Energy Data Science Take-Home Assignment: REFIT Data Cleaning and Appliance Disaggregation

An end-to-end, reproducible Data Science pipeline for Non-Intrusive Load Monitoring (NILM) and smart meter data cleaning using the **REFIT Electrical Load Measurements** dataset from the University of Strathclyde.

This project delivers:
1. **Interactive Jupyter Notebooks (.ipynb)** implementing data cleaning, exploratory analysis, cycle detection, feature engineering, and appliance disaggregation.
2. **Official REFIT Data Ingestion**: Clean handling of both raw 8-second telemetry (`REFIT_RAW_081116.7z`) and official cleaned records (`CLEAN_REFIT_081116.7z`).
3. **Comprehensive Technical Documentation**: [`DATA.md`](file:///c:/Users/hitan/OneDrive/Desktop/flock_energy_assign/DATA.md), [`RESULTS.md`](file:///c:/Users/hitan/OneDrive/Desktop/flock_energy_assign/RESULTS.md), and [`Recommendation.md`](file:///c:/Users/hitan/OneDrive/Desktop/flock_energy_assign/Recommendation.md).

---

## 1. Project Architecture & Notebook Structure

The analysis is structured into three self-contained, executed Jupyter Notebooks following standard Data Science lifecycle workflows:

```
flock_energy_assign/
├── 01_Data_Cleaning.ipynb               # Task 1: Raw data inspection, deduplication, outage handling, 1-min resampling
├── 02_Washing_Machine_EDA.ipynb         # Task 2: Multi-home EDA, physical cycle detection, diurnal & weekly patterns
├── 03_Appliance_Disaggregation.ipynb    # Task 3: Contextual feature engineering, Random Forest NILM model vs. baselines
├── notebooks/                           # Mirrored copies of the notebooks
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_Washing_Machine_EDA.ipynb
│   └── 03_Appliance_Disaggregation.ipynb
├── data/
│   ├── raw/                             # Extracted raw files (RAW_House1_Part1.csv, Part2.csv)
│   ├── clean/                           # Extracted cleaned files (CLEAN_House1, 2, 3, 20.csv)
│   └── processed/                       # 1-minute resampled feature sets
├── figures/                             # High-resolution publication plots
│   ├── cleaning/                        # Before/after raw vs. cleaned comparison traces
│   ├── eda/                             # 24h traces, cycle anatomies, cross-house comparison
│   └── model/                           # Actual vs. predicted traces, error residuals, feature importances
├── src/                                 # Modular Python utilities
│   ├── config.py                        # Household metadata, channel mapping, thresholds
│   ├── cleaning.py                      # Data validation and cleaning routines
├── README.md                            # Setup, methodology, trade-offs, AI disclosures
├── DATA.md                              # Data quality audit, cleaning rules, gap analysis
├── RESULTS.md                           # EDA findings, model benchmarks, failure diagnostics
├── Recommendation.md                    # Utility executive summary & Section 4 policy answers
└── requirements.txt                     # Pinned Python package dependencies
```

---

## 2. Notebook Walkthroughs & Requirements Coverage

### [01_Data_Cleaning.ipynb](file:///c:/Users/hitan/OneDrive/Desktop/flock_energy_assign/01_Data_Cleaning.ipynb) (Requirement 1)
- **Target Household:** House 1 (Detached, 4-bed, 2 occupants).
- **Telemetry Inspected:** 300,000 raw readings at ~8-second nominal sampling.
- **Timestamp Validation:** Evaluates UTC timestamps, chronological monotonicity, and retrograde jumps.
- **Duplicate Handling:** Detects zero-delta duplicates (~0.3% of rows) caused by simultaneous sensor broadcasts. Applies a stated, energy-conserving rule: *group by timestamp and calculate arithmetic mean power*.
- **Outlier Handling:** Clamps negative noise (< 0 W) to 0 W and caps unphysical spikes (> 15 kW aggregate, > 4 kW sub-meters).
- **Resampling & Outage Strategy:** Resamples to regular 1-minute intervals. Short gaps ($\le 3$ min) are linearly interpolated; extended gaps (> 3 min) are strictly flagged as `is_outage=True` and kept as NaN rather than interpolated without justification.
- **Visualisation:** 48-hour before-and-after plot comparing raw irregular readings vs. cleaned 1-minute signals for Aggregate, Appliance 5 (Washing Machine), and Appliance 6 (Dishwasher).

### [02_Washing_Machine_EDA.ipynb](file:///c:/Users/hitan/OneDrive/Desktop/flock_energy_assign/02_Washing_Machine_EDA.ipynb) (Requirement 2)
- **Target Households:** Houses 1, 2, 3, and 20.
- **Physical Cycle Detection Algorithm:** Formulates a plain-language, multi-stage detection rule:
  1. $P \ge 15\text{ W}$ marks active state (filtering 1–5 W standby).
  2. Max pause tolerance $\le 3\text{ minutes}$ below 15 W bridges internal wash/soak phases into the same cycle.
  3. Minimum duration $\ge 15\text{ minutes}$ filters accidental plug toggles.
  4. Validation check: Peak power $\ge 500\text{ W}$ or energy $\ge 0.05\text{ kWh}$ confirms water-heating element operation.
- **Cycle Metrics:** Extracts duration, peak power, average active power, and energy per cycle (kWh).
- **Temporal Routines:** Identifies pronounced morning load clustering (08:00–11:00) and weekend concentration (Saturday/Sunday).
- **Visualisations:**
  - *(i) Representative 24-Hour Trace:* House 1 aggregate load vs. washing machine event.
  - *(ii) Detected Cycle Anatomy:* Zoomed multi-phase profiles showing heating, tumble wash, rinse, and spin extraction.
  - *(iii) Cross-Household Comparison:* Boxplots of cycle duration, peak power, and energy across all 4 homes.

### [03_Appliance_Disaggregation.ipynb](file:///c:/Users/hitan/OneDrive/Desktop/flock_energy_assign/03_Appliance_Disaggregation.ipynb) (Requirement 3)
- **Problem Formulation:** Estimating 1-minute washing machine power $\hat{y}_t$ from whole-home aggregate $P_{agg}(t)$.
- **Feature Engineering:** Builds 28 contextual features:
  - Symmetrical lags and leads: $t \pm 1, 2, 3, 5, 10, 15$ minutes.
  - Multi-scale rolling statistics: mean, std, min, max over 5, 15, 30, and 60 minutes.
  - Transient step delta: $\Delta P_t = P_t - P_{t-1}$.
  - Cyclical diurnal and weekly encodings ($\sin/\cos$ of hour and day of week).
- **Leakage-Free Validation Strategy:** **Out-of-Household Split** (train on Houses 1, 2, 3; evaluate on completely unseen House 20). Prevents synthetic temporal leakage and tests true generalization to new utility customers.
- **Baselines:** Always-Off Baseline ($\hat{y} = 0\text{ W}$) and Mean-Profile Baseline.
- **Model:** Random Forest Regressor (100 trees, regularized tree depth).
- **Evaluation:** Evaluates MAE, RMSE, $R^2$, Relative Energy Error (SAE), and active-state classification metrics (Precision, Recall, F1-Score).
- **Visualisations & Diagnostics:** Multi-day actual vs. predicted trace overlay on unseen House 20, error residual distribution, and feature importance ranking. Detailed failure analysis of resistive heating cross-talk and baseload masking.

---

## 3. Setup and Execution Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows / macOS / Linux

### Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-repo/flock_energy_assign.git
cd flock_energy_assign
pip install -r requirements.txt
```

### Running the Notebooks
Launch Jupyter Lab or Notebook:
```bash
jupyter lab
```
Execute the notebooks in sequence:
1. `01_Data_Cleaning.ipynb`
2. `02_Washing_Machine_EDA.ipynb`
3. `03_Appliance_Disaggregation.ipynb`

### Headless Execution (Automated Pipeline)
To re-execute all three notebooks headlessly from the command line:
```bash
jupyter nbconvert --to notebook --execute 01_Data_Cleaning.ipynb
jupyter nbconvert --to notebook --execute 02_Washing_Machine_EDA.ipynb
jupyter nbconvert --to notebook --execute 03_Appliance_Disaggregation.ipynb
```

---

## 4. Key Methodological Decisions & Trade-Offs

| Decision | Approach Adopted | Alternative Considered | Rationale / Trade-Off |
| :--- | :--- | :--- | :--- |
| **Resampling Resolution** | 1-minute mean power | 8-second raw or 15-minute intervals | 1-minute aligns with typical smart sub-metering, filters high-frequency sensor noise, preserves the distinct $\approx 15$-min heating surge, and is computationally tractable. |
| **Outage Imputation** | Interpolate $\le 3$ min; flag $> 3$ min as NaN | Mean imputation or forward-filling multi-hour gaps | Forward-filling large multi-hour disconnects fabricates active consumption, distorting baseline energy balances. |
| **Validation Design** | Split by Household (train: H1, 2, 3; test: H20) | Random train/test split or k-fold shuffle | Random shuffling across time creates massive temporal leakage via sliding lag/lead windows. Household-level splitting reflects true real-world utility deployment. |
| **Model Choice** | Contextual Random Forest Regressor | Deep Seq2Point (CNN) or Point-by-point Linear | Random Forest handles non-linear interactions, resists overfitting, requires zero GPU overhead, and runs reproducibly on any standard machine. |
| **Evaluation Metrics** | Dual Metric Suite: MAE/RMSE + F1/Recall | MAE / RMSE only | Because washing machines are off $>98\%$ of the time, an Always-Off baseline yields deceptively low MAE ($8.9\text{ W}$) despite $0\%$ energy recall. F1 and energy error are essential. |

---

## 5. Academic References & Citations

1. **Murray, D., Liao, J., Stankovic, L., Stankovic, V., et al. (2017).** *A data management platform for personalised real-time energy feedback.* IEEE Transactions on Consumer Electronics, and Scientific Data 2016 REFIT release.
2. **Kelly, J., & Knottenbelt, W. (2015).** *Neural NILM: Deep neural networks applied to energy disaggregation.* Proceedings of the 2nd ACM International Conference on Embedded Systems for Energy-Efficient Built Environments (BuildSys), pp. 55–64.
3. **Batra, N., Kelly, J., Parson, O., et al. (2014).** *NILMTK: An open source toolkit for non-intrusive load monitoring.* Proceedings of the 5th International Conference on Future Energy Systems (e-Energy), pp. 265–276.
4. **Stankovic, L., Stankovic, V., Liao, J., & Murray, D. (2016).** *Measuring the energy vulnerability of households through smart meter data.* IEEE Transactions on Smart Grid, 7(4), 2185–2193.

---

## 6. Disclosure of AI Tools Used

In accordance with transparent academic and engineering practices:
- **LLM Assistance:** An advanced coding assistant was utilized as a pair-programming collaborator for drafting boilerplate notebook scaffolds, verifying mathematical metric definitions, and generating initial Markdown templates.
- **Independent Validation:** All data loading, cleaning scripts, physical cycle detection logic, model training, metric computations, and plot generation were executed directly against the real Strathclyde REFIT dataset and verified for reproducibility.
