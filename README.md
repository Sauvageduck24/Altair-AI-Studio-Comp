# 📘 EUR/USD Trading Pattern Analysis with AI Studio & Streamlit

### Altair Global Student Contest 2025 — University of Málaga

---

## 🚀 Define your Use Case

In financial markets, high-frequency EUR/USD trading data contains both *structure* (market regimes, volatility clusters) and *noise* (anomalous price behaviors).
Our goal is to leverage **AI Studio (RapidMiner)** to automatically **detect and classify these regimes and anomalies**, creating a foundation for adaptive trading strategies.

This project explores how **unsupervised learning** —through *PCA dimensionality reduction*, *Outlier Detection*, and *Clustering*— can transform raw financial data into actionable trading insights.
The resulting dataset and visual dashboard allow traders and analysts to:

* 🧭 Identify *non-stationary regimes* and volatility shifts.
* ⚠️ Detect *outlier candles* (abnormal market events).
* 🎯 Design *decision policies* (e.g., “avoid trading during outlier events” or “treat clusters differently”).

---

## 🧩 Process Overview

### **1. Data Preparation**

* **Dataset:** EUR/USD 15-minute OHLCV data.
* **Pre-processing (AI Studio):**

  * Remove *date column* (for time-agnostic modeling).
  * Add *unique ID column* for traceability.
  * Replace *missing values* with appropriate imputation.
  * Apply *Z-Transformation* for normalization (zero mean, unit variance).

### **2. Feature Extraction**

* **Dimensionality Reduction:** PCA (Principal Component Analysis) tested with different component settings to retain key variance directions.
* **Outlier Detection:** Multiple methods (e.g., Isolation Forest, LOF, PCA-based Outlier Detection) were compared to flag abnormal market behavior.
* **Clustering:** Applied clustering (e.g., K-Means, DBSCAN) to group market states into interpretable clusters.

### **3. Data Integration**

* Combined outputs into a single enriched dataset:

  * `cluster` → market regime label
  * `outlier_flag` → anomaly indicator
* This enhanced dataset forms the analytical base for the visualization dashboard.

### **4. Visualization & Insights**

* A custom **Streamlit Web Application** (`app.py`) was built to visualize the processed dataset:

  * Interactive **candlestick chart** with clusters and outliers over time.
  * Multiple *display modes* (vertical lines, markers, or filtered outliers only).
  * **Statistics panel** summarizing cluster distributions and outlier density.
  * **Timeline view** showing how clusters evolve across market sessions.

### **5. Testing & Iteration**

* Verified dataset consistency and synchronization between raw and processed data.
* Evaluated interpretability of clusters (e.g., volatility vs. calm regimes).
* Tested dashboard usability for different parameter combinations.

---

## 💡 Results & Insights

| Metric                      | Description                                        | Example Insight                                                              |
| --------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Outlier Ratio**           | % of candles flagged as anomalies                  | 3–5% of candles show abnormal price ranges                                   |
| **Cluster Separation**      | PCA visualization shows strong regime structure    | Distinct low-volatility vs. breakout regimes                                 |
| **Visual Interpretability** | Cluster colors clearly highlight structural shifts | “Blue clusters” dominate calm periods; “red clusters” align with news events |
| **Decision Framework**      | Strategy-level implications                        | Possible rule: *avoid trading during outlier candles in Cluster 3*           |

---

## 🌍 Real-World Relevance & Originality

Financial time series are inherently noisy and regime-dependent.
Most retail strategies ignore this and treat all periods equally — leading to poor robustness.
Our project demonstrates an **original application** of unsupervised ML to enhance **decision awareness** in trading systems by exposing *hidden structure* and *market anomalies*.

By integrating **AI Studio workflows** and a **Streamlit visualization layer**, we bridge the gap between *data science experimentation* and *real-time decision support*.

---

## 🧠 Workflow Design Quality

| Phase             | Tool / Technique                     | Purpose                  |
| ----------------- | ------------------------------------ | ------------------------ |
| Data Cleaning     | RapidMiner (Replace Missing Values)  | Ensure integrity         |
| Normalization     | Z-Transformation                     | Scale features uniformly |
| PCA               | RapidMiner (PCA Operator)            | Reduce redundancy        |
| Outlier Detection | Isolation Forest / LOF / PCA-Outlier | Identify anomalies       |
| Clustering        | K-Means / DBSCAN                     | Segment market regimes   |
| Visualization     | Streamlit + Plotly                   | Deliver interpretability |

---

## 📊 Clarity of Insights

The Streamlit dashboard (`app.py`) transforms complex ML outputs into intuitive financial visualization:

* Dynamic candlesticks per cluster color.
* Real-time filtering by cluster or anomaly.
* Statistical summaries (outlier % per cluster).
* Clear legends and intuitive controls for non-technical users.

This allows users to explore hypotheses interactively, fostering **data-driven trading intuition**.

---

## 🧭 Decision Support

The project supports multiple decision paradigms:

* **Risk-Aware Mode:** Avoid entering trades during outlier-flagged candles.
* **Cluster-Adaptive Mode:** Apply strategy variants depending on cluster regime.
* **Exploratory Mode:** Investigate anomalies for research or model retraining.

Ultimately, the system enables **AI-assisted situational awareness** — turning raw data into *strategic foresight*.

---

## 🌟 Impact & Next Steps

This project provides a framework that can be expanded to:

* Integrate **real-time data feeds** for live anomaly detection.
* Extend to other instruments (GBP/USD, XAU/USD, SPX500).
* Feed results into **reinforcement learning** agents (e.g., Qlib, RD-Agent) for dynamic decision optimization.
* Build **backtesting modules** connecting detected regimes to profitability metrics.

**Impact:**
By combining explainable AI with interactive visualization, this system empowers traders, researchers, and students to *understand* markets — not just predict them.

---

## 🧰 Technical Stack

| Component                       | Technology                    |
| ------------------------------- | ----------------------------- |
| Data Processing                 | Altair AI Studio (RapidMiner) |
| Visualization                   | Streamlit + Plotly            |
| Backend                         | Python 3.10                   |
| Data                            | EUR/USD 15M OHLCV dataset     |
| PCA, Clustering, Outlier Models | AI Studio Operators           |
| Hosting (optional)              | Streamlit Cloud / Local       |

---

## 📎 Repository Structure

```
├── app.py                         # Streamlit visualization dashboard
├── data_raw/
│   └── EURUSD_15M.csv             # Original OHLCV data
├── data_preprocessed/
│   └── prueba_results_comp.csv    # AI Studio processed dataset (PCA, clusters, outliers)
├── README.md                      # Project documentation
```

---

## 🏁 Authors & Acknowledgements

