# 📊 Anomaly & Clustering Workflow for Market Patterns
**Altair Global Student Contest 2025 – Project Submission**

## 📝 Overview
This project explores how to detect and characterize anomalies in financial time series
(OHLC candles and derived indicators) using combined pipelines of outlier detection and
clustering. The core idea is to chain these methods in different orders to maximize
signal quality and extract actionable insights about market regimes and rare events. In
addition, we provide an optional backtesting module to evaluate the practical impact of
signals on simple trading strategies.

## 🎯 Objectives
- Detect market anomalies in time series data.
- Discover regime patterns and their relationship with anomalies.
- Assess the practical usefulness and consistency of signals.
- Provide a clear, reproducible, and impactful workflow for the contest.

## 📦 Optional: Strategy Backtesting
Leverages anomaly and clustering outputs to simulate simple strategies and quantify
impact.

- Inputs: per-timestamp signals (e.g., anomaly scores/flags, cluster labels, regime
  transitions) and forward returns from the dataset.
- Rules (examples):
  - Buy (or reduce short) following specific anomaly types or regime transitions.
  - Avoid trading during high-risk anomaly clusters; re-enter on normalization.
  - Event-study aggregation into executable rules with holding windows t+{1,4,8,24}.
- Metrics:
  - CAGR, annualized volatility, Sharpe/Sortino
  - Max drawdown, Calmar, hit ratio
  - Turnover, average holding time, slippage sensitivity
- Outputs:
  - Equity curve and drawdown plots
  - Per-regime performance table and anomaly-condition attribution
  - Sensitivity analyses across thresholds and holding windows

## 🔄 Workflows Implemented
We implement four main variants:

1. **Outliers → Cluster (with outliers)**  
   - Detect global outliers with LOF/Isolation Forest.  
   - Cluster the full dataset (including outliers) to observe how rare and normal
     samples group in the same space.

2. **Outliers → Cluster (without outliers)**  
   - Filter and remove global outliers first.  
   - Cluster only "normal" points to study the structure without noise.

3. **Cluster → Outliers (per-cluster)**  
   - Segment the dataset into clusters (market regimes).  
   - Detect local outliers within each cluster using LOF to capture anomalies relative
     to each regime’s neighborhood.

4. **Cluster → No Outliers (baseline)**  
   - Keep only non-anomalous observations inside each cluster.  
   - Serves as a control to measure differences in returns, volatility, or patterns.

## ⚙️ Technical Setup
- **Input**: Preprocessed CSV/Parquet under `data_preprocessed/`.  
  Example: `data_preprocessed/FUNDEDNEXT_EURUSD_15_2000-01-01_2025-01-01__preprocessed.csv`
- **Raw data**: `data_raw/` (e.g., `FUNDEDNEXT_EURUSD_15_2000-01-01_2025-01-01.csv`).
- **Preprocessing script**: `data_preprocess.py` (optional local preprocessing).
- **Features**: OHLC candles, volume, technical indicators, and statistical features
  (volatility, entropy, Hurst exponent, etc.).
- **Tools**: Altair AI Studio (predict/outliers/clustering operators).
- **Validation**: Time-based walk-forward (train/validation/test).
- **Metrics**:
  - Percentage of outliers per cluster
  - Cluster stability with and without outliers
  - Average returns after anomalies (event-study at t+{1,4,8,24})
  - Consistency between LOF and Isolation Forest

## 📈 Expected Insights
- Global outliers are not always local outliers: some events are rare globally, others
  only within a regime.
- Clustering after removing outliers reveals a cleaner market structure, helpful for
  modeling.
- Clustering only outliers can uncover types of anomalies (e.g., volatility shocks,
  session gaps, extreme jumps).
- Derived signals: recurrent anomaly patterns may imply short-term positive/negative
  return bias.

## 📊 Deliverables
- Four workflows in Altair AI Studio (the variants above).
- Dashboards/plots:
  - Anomaly heatmap (day × hour)
  - LOF/IF score distributions
  - Cluster visualizations with and without outliers
  - Event-study: returns following anomaly events
- Backtesting artifacts (optional): equity curves, performance tables, and sensitivity
  analyses.
- Short video walkthrough covering:  
  1) Problem → data → features  
  2) Combined workflows (outliers/cluster)  
  3) Key insights  
  4) Impact: how to use signals for decision support

---

## 🚀 How to Run
1. (Optional) Preprocess locally:  
   - Place raw data under `data_raw/`.  
   - Run `data_preprocess.py` to create preprocessed files under `data_preprocessed/`.
2. Import a preprocessed file (CSV/Parquet) into Altair AI Studio.
3. Execute the four workflows. Use macros for parameters such as `k`, `eps`,
   `contamination`, and thresholds.
4. Export results (scores, labels, clusters).
5. Generate plots and summary tables (optional notebooks or your preferred tooling).
6. (Optional) Run strategy backtests using exported signals and forward returns.
7. Prepare dashboards and a short video summarizing the findings and, if used,
   the backtesting impact.

---

## 🙌 Acknowledgments
- Built for the
  [Altair Global Student Contest 2025](https://web.altair.com/global-student-contest-2025).

## 👥 Authors (Group)
- Esteban Sánchez  
- Pablo Delgado
