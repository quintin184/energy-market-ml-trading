# Machine Learning Trading Strategy for the Irish Single Electricity Market

**Tools:** Python, Scikit-learn, XGBoost, Pandas  
**Data:** ISEM market data 2022–2025, provided by Energia Group

---

## Overview

This project develops a machine learning algorithmic trading strategy for the Irish Single Electricity Market (ISEM). 
The ISEM is a shared energy market across Ireland and Northern Ireland,
designed to supply electricity to homes and businesses at a fair price through competitive bidding between generators.

The project was completed as part of a data challenge set by Energia Group through my college. 
The goal was to work through a high-dimensional dataset covering multiple trading windows, 
isolate the most useful features, and build a strategy that maximises profit per megawatt hour (€/MWh) while managing risk.

---

## The Market

The ISEM consists of 5 trading windows:

- **DAM** — Day-Ahead Market: prices set for every 30-minute interval the day before delivery
- **IDA1** — First Intraday Auction: opens after DAM closes, allows position adjustment closer to real-time
- **IDA2** — Second Intraday Auction: runs on the day of delivery
- **IDA3** — Third Intraday Auction: runs from 17:00 to 23:00 on delivery day
- **BM** — Balancing Market: real-time and most volatile of all windows

After data cleaning, IDA2 and IDA3 were removed due to structural missing values (limited operating hours). All modelling focused on the **DAM → BM** and **DAM → IDA1** trading routes.

---

## Approach

### Data Preparation
- Converted raw Parquet files to CSV and cleaned chronologically
- Removed real-time variables to prevent information leakage
- Engineered two additional features: residual demand (demand minus renewables) and wind forecast uncertainty
- Training set: 01/01/2022 – 31/12/2024 | Test set: 01/01/2025 – 31/12/2025

### Target Variable
Rather than predicting price magnitude, the models predicted **price direction** — whether the later market price would be higher or lower than the DAM price. This classification target (y_dir: +1 or –1) allowed the strategy to decide whether to buy or sell at each time step.

### Models Tested
- **Random Forest** — ensemble of decision trees trained on random data subsets
- **XGBoost** — sequential gradient boosting, each tree correcting errors of the previous
- **Logistic Regression** — probability-based classification using the sigmoid function

---

## Key Results

### Basic Model (No Restrictions) — DAM → BM

| Model | Accuracy | No. Trades | Profit (€) | €/MWh |
|---|---|---|---|---|
| Random Forest | 0.555 | 17,516 | 63,245 | 3.61 |
| XGBoost | 0.557 | 17,516 | 96,898 | 5.53 |
| Logistic Regression | 0.549 | 17,516 | 106,472 | 6.08 |

DAM → BM consistently outperformed DAM → IDA1 across all models, likely due to the greater price volatility over the longer time gap.

### With Confidence Threshold (Top 4.2% of Trades)

| Model | Accuracy | No. Trades | Profit (€) | €/MWh |
|---|---|---|---|---|
| Logistic Regression | 0.549 | 2,047 | 41,758 | 20.4 |

By filtering to only the model's most confident predictions, €/MWh increased from 6.08 to 20.4 — a 3x improvement.

### High Wind Uncertainty + 1% Confidence Threshold

| Model | Accuracy | No. Trades | Profit (€) | €/MWh |
|---|---|---|---|---|
| Logistic Regression | 0.790 | 214 | 15,272 | 71.37 |

Focusing exclusively on periods of high wind forecast uncertainty (Ireland's Atlantic weather makes wind hard to predict) and restricting to only the top 1% of confident trades delivered the strongest €/MWh of any strategy tested.

---

## Key Findings

- The DAM → BM route is significantly more profitable than DAM → IDA1 due to greater price spreads
- Logistic Regression outperformed more complex models, suggesting the DAM–BM price relationship is largely linear
- Higher model accuracy does not always mean higher profit — a phenomenon known as the Accuracy Paradox
- Wind forecast uncertainty is a strong signal: the most profitable trades clustered in high-uncertainty periods, particularly in Q1 and Q4 when Irish wind conditions are most volatile
- Applying a confidence threshold dramatically improves €/MWh at the cost of total trade volume

---

## Repository Contents

- `AnalythonProject_final.ipynb` — full analysis notebook including data cleaning, feature engineering, model training and results
- `energy-market-write-up.pdf` — written report with figures, tables and conclusions

---

## References

- Nvidia. *What is XGBoost?* https://www.nvidia.com/en-gb/glossary/xgboost/
- Mostafa Shams. *A Statistical Evaluation of Machine Learning Profitability* https://www.qeios.com/read/7A61RS
- Rishabh Singh. *Logistic Regression* https://medium.com/@RobuRishabh/logistic-regression-c2d2bac7afd8
