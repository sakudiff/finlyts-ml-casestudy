# Hybrid Econometric and Machine Learning Modeling of Philippine Stock Returns

**Course:** FINLYTS
**Professor:** Bobby Baylon Jr.
**Status:** In Progress — Company selection pending

---

## Project Overview

This project develops and compares econometric and machine learning models to
predict multi-horizon stock returns of a selected Philippine Stock Exchange (PSE)
listed company. The analysis spans three prediction horizons and five model
families, evaluated on both statistical and financial performance metrics.

---

## Paper Outline

The final deliverable is a full academic paper structured as follows.

**Section 1: Introduction**
Justification for the selected stock pair, rationale for the model choices,
and the specific research question being answered within the context of
Philippine capital markets.

**Section 2: Literature Review**
Survey of prior work on ML-based stock return prediction, the Efficient Market
Hypothesis debate in frontier markets, and PSE-specific empirical studies.

**Section 3: Data and Feature Engineering**
Data sources and sample period (2019 to present). Construction of all three
target variables with explicit formulas. All mandatory technical indicators
(MA, EMA, RSI, MACD, rolling volatility), cross-asset features (PSEi, competitor
returns, macro variable), feature transformations (interaction term, squared
volatility), and the COVID regime dummy. Descriptive statistics table.

**Section 4: Methodology**
Time-series split rationale. OLS benchmark specification. Architecture and
training procedure for each ML model (Random Forest, SVM, XGBoost, ANN).
Hyperparameter tuning procedure. Definition of all evaluation metrics (RMSE,
MAE, R-squared, Directional Accuracy).

**Section 5: Results**
Main results tables: RMSE, MAE, R-squared, and Directional Accuracy across
all five models and all three prediction horizons, forming a 5 x 3 matrix per
metric. Feature importance plots for Random Forest and XGBoost. OLS coefficient
table. Predicted vs. actual plot for the best-performing model.

**Section 6: Discussion**
Economic interpretation of all findings. Model comparison with financial
reasoning. Horizon analysis (why short-horizon may differ from long-horizon
predictability). Feature importance interpretation in terms of momentum, trend,
and macro channels. COVID structural break effects on coefficients. Assessment
of whether results support or challenge weak-form EMH for the PSE.
Overfitting analysis comparing training and testing performance.

**Section 7: Conclusion and Recommendations**
Summary of findings, practical implications for analysts, limitations of the
study, and directions for future research.

**Section 8: References**
All citations in APA 7 format with DOI or URL.

**Section 9: Appendix**
Full reproducible R code. All scripts are numbered 01 through 06 and run
sequentially from a clean R environment.

---

## Models

| Model | Type | Package |
|---|---|---|
| OLS | Benchmark econometric | `stats::lm` |
| Random Forest | Ensemble ML | `ranger` via `tidymodels` |
| Support Vector Machine | Kernel ML | `kernlab` via `tidymodels` |
| XGBoost | Gradient boosting | `xgboost` via `tidymodels` |
| ANN | Neural network | `nnet` via `tidymodels` |

---

## Prediction Targets

| Horizon | Formula |
|---|---|
| 1-day | $r_{t+1} = \log(P_{t+1} / P_t)$ |
| 3-day cumulative | $r_{t:t+3} = \sum_{i=1}^{3} r_{t+i}$ |
| 5-day cumulative | $r_{t:t+5} = \sum_{i=1}^{5} r_{t+i}$ |

Separate models are estimated for each target. No look-ahead bias.

---

## Repository Structure

```
├── data/
│   ├── raw/              # Pulled directly from Yahoo Finance via tidyquant
│   └── processed/        # Feature-engineered analysis-ready dataset
├── R/
│   ├── 01_data_pull.Rmd
│   ├── 02_feature_engineering.Rmd
│   ├── 03_ols_baseline.Rmd
│   ├── 04_models_rf_xgboost.Rmd
│   ├── 05_models_svm_ann.Rmd
│   └── 06_evaluation.Rmd
├── paper/
│   ├── main.tex          # Final LaTeX paper
│   ├── references.bib    # APA 7 BibTeX entries
│   ├── tables/           # Auto-generated LaTeX tables from R
│   └── figures/          # Auto-generated plots from R
├── docs/
│   └── MACHINE_LEARNING_CASE_STUDY_INSTRUCTIONS_AND_RUBRIC.pdf
├── output/
│   ├── models/           # Saved model objects (.rds)
│   └── predictions/      # Test set predictions (.csv)
├── .gitignore
└── README.md
```

---

## Role Assignments

| Person | Primary Sections | RMD Scripts |
|---|---|---|
| Person 1 | Section 3 (data), Appendix | 01_data_pull.Rmd |
| Person 2 | Section 2, Section 4 (writing), References | 02_feature_engineering.Rmd (co-owner) |
| Person 3 | Section 5 | 02 (co-owner), 03, 04, 06 |
| Person 4 | Section 1, Section 6, Section 7 | 05_models_svm_ann.Rmd |

---

## Critical Rules

1. Time-series split only. `train_test_split(shuffle=TRUE)` equivalent in R is forbidden.
2. No look-ahead bias. All features use `dplyr::lag()`. Targets use `dplyr::lead()` and are assigned as separate columns.
3. SVM requires feature standardization. Scaler fitted on training set only.
4. Hyperparameter tuning uses `rsample::rolling_origin()` inside `tune::tune_grid()`, not standard k-fold CV.
5. All citations APA 7 with DOI or URL.

---

## Data Sources

All data pulled programmatically via `tidyquant::tq_get()` from Yahoo Finance.

| Series | Yahoo Finance Ticker |
|---|---|
| Target stock | TBD.PS |
| Competitor stock | TBD.PS |
| PSEi Index | ^PSEi |
| USD/PHP Exchange Rate | PHP=X |
| Bitcoin | BTC-USD |
| WTI Crude Oil | CL=F |

**Note:** Company tickers will be confirmed and updated once the group finalizes
stock selection. BDO.PS vs BPI.PS (banking sector) is the recommended default.

---

## Instructions and Rubric

See `docs/MACHINE_LEARNING_CASE_STUDY_INSTRUCTIONS_AND_RUBRIC.pdf`

---

## Bonus Targets (optional, +5 pts)

- Rolling or expanding window validation via `rsample::rolling_origin()`
- Simple trading strategy evaluation based on Directional Accuracy
- Hybrid econometric plus ML residual model
- SHAP interpretability via `SHAPforxgboost`
