# MACHINE LEARNING CASE STUDY INSTRUCTIONS

**Title: Hybrid Econometric and Machine Learning Modeling of Philippine Stock Returns**

**COURSE: FINLYTS**  
**PROFESSOR: BOBBY BAYLON JR.**

---

### Objective
You are required to develop and compare **econometric and machine learning models** to predict **multi-horizon stock returns** of a selected Philippine-listed company.

Specifically, you will:

*   **Forecast:**
    *   1-day ahead return (t+1)
    *   3-day cumulative return (t:t+3)
    *   5-day cumulative return (t:t+5)
*   **Compare:**
    *   Linear econometric model (OLS)
    *   Machine learning models (RF, SVM, XGBoost, ANN)
*   **Evaluate:**
    *   Predictive performance
    *   Economic significance
    *   Model interpretability

---

### 📊 1. Asset Selection
You must select:

*   One **PSE-listed company** (target asset)
*   One **competitor firm** (same sector/industry)

**Requirements:**
*   Provide justification based on:
    *   Liquidity
    *   Sector relevance
    *   Economic importance

---

### 📅 2. Data Requirements
**Time Frame:**
*   Minimum: **7+ years of daily data**
*   Recommended: **2019–current**

**Required Data:**
*   Target stock (price and volume)
*   Competitor stock
*   PSEi Index
*   At least **one global/macroeconomic variable**:
    *   Oil prices
    *   Cryptocurrency (e.g., Bitcoin)
    *   Exchange rate (USD/PHP)

---

### 🎯 3. Target Variables (Multi-Horizon)
You must construct the following:

*   **1-day ahead return:**
    $$r_{t+1} = \log\left(\frac{P_{t+1}}{P_t}\right)$$

*   **3-day cumulative return:**
    $$r_{t:t+3} = \sum_{i=1}^{3} r_{t+i}$$

*   **5-day cumulative return:**
    $$r_{t:t+5} = \sum_{i=1}^{5} r_{t+i}$$

👉 You must estimate **separate models for each target**.

---

### ⚙️ 4. Feature Engineering

#### A. Mandatory Technical Indicators
You are required to include the following:

**Trend**
*   MA(5), MA(10), MA(20)
*   EMA(12), EMA(26)
*   At least one MA ratio (e.g., MA5 / MA20)

**Momentum**
*   Lagged returns (1, 2, 3 days)
*   Momentum (5-day, 10-day)
*   RSI (14)
*   MACD and signal line

**Volatility**
*   Rolling standard deviation (5-day, 10-day)
*   Volatility ratio (short-term / long-term)

#### B. Market & Cross-Asset Features (REQUIRED)
*   PSEi returns (current and/or lagged)
*   Competitor stock returns
*   At least **one** of the following:
    *   Oil returns
    *   Cryptocurrency returns (e.g., BTC)
    *   Exchange rate returns (USD/PHP)

#### C. Feature Transformations (REQUIRED)
Include at least **one**:
*   Interaction term (e.g., momentum × volatility)
*   Nonlinear transformation (e.g., squared volatility)

#### D. Regime Variable (REQUIRED)
Include at least **one**:
*   COVID period dummy
    **or**
*   High-volatility regime indicator

⚠️ **CRITICAL RULE: NO LOOK-AHEAD BIAS**
All features must be constructed using **only information available at time t or earlier**. Any violation (e.g., using future data) will result in **automatic failure**.

---

### 🔄 5. Data Splitting (STRICT REQUIREMENT)
You must use a **time-series split**:
*   Training set: first 70–80% of observations
*   Test set: remaining 20–30%

❌ Random splitting is **NOT** allowed.

---

### 🤖 6. Models to Estimate
You must estimate the following:
1.  **OLS (Benchmark Model)**
2.  **Random Forest (RF)**
3.  **Support Vector Machine (SVM)**
4.  **XGBoost**
5.  **Artificial Neural Network (ANN)**

⚙️ **Hyperparameter Tuning (REQUIRED)**
You must tune **at least one ML model** and explain your choices.

---

### 📈 7. Model Evaluation

#### A. Statistical Metrics
*   RMSE
*   MAE
*   R²

#### B. Financial Metrics (REQUIRED)
*   **Directional Accuracy (DA)**

#### C. Benchmark Comparison (REQUIRED)
Compare against a **naïve model**, such as:
*   Previous return (lagged return)

---

### 🧠 8. Required Analysis

#### A. Model Comparison
*   Which model performs best?
*   Does ML outperform OLS?

#### B. Horizon Analysis
*   Compare performance across:
    *   1-day vs 3-day vs 5-day predictions

#### C. Feature Importance
*   OLS: coefficient interpretation
*   ML: feature importance

#### D. Economic Interpretation
*   Are results financially meaningful?
*   Do findings align with theory?

#### E. Overfitting Assessment
*   Compare training vs testing performance

---

### 📄 9. Paper Structure
Your paper must include:
1.  Introduction
2.  Literature Review
3.  Data and Feature Engineering
4.  Methodology
5.  Results
6.  Discussion
7.  Conclusion and Recommendations
8.  References
9.  Appendix (Code)

---

### ⭐ 10. Bonus (Optional but Highly Rewarded)
*   Rolling/expanding window validation
*   Trading strategy evaluation
*   Hybrid econometrics + ML model
*   Advanced interpretability (e.g., SHAP)

---

### 🎯 FINAL NOTES (IMPORTANT FOR STUDENTS)
*   This is **NOT** a coding exercise.
*   This is a **financial modeling and interpretation exercise**.
*   Emphasis is placed on:
    *   Proper methodology
    *   Economic reasoning
    *   Clear interpretation

---

## GRADING RUBRIC (100 POINTS)

### 1. Data & Feature Engineering (20 pts)
**What you are evaluating:**
*   Correct construction of targets (t+1, t+3, t+5)
*   Proper feature engineering (technical + macro)
*   No look-ahead bias

**Guide:**
*   **18–20 (Excellent):** Complete, correct, no leakage, strong features
*   **14–17 (Good):** Minor issues, mostly correct
*   **10–13 (Fair):** Missing key features or weak construction
*   **0–9 (Poor):** Incorrect targets or evidence of leakage

### 2. Modeling & Methodology (20 pts)
**What you are evaluating:**
*   Correct implementation of all models
*   Proper time-series split
*   At least one tuned ML model

**Guide:**
*   **18–20:** All models correct + tuning + proper validation
*   **14–17:** Minor issues (e.g., weak tuning)
*   **10–13:** Missing models or incorrect split
*   **0–9:** Major methodological flaws

### 3. Results & Evaluation (20 pts)
**What you are evaluating:**
*   Use of RMSE, MAE, R²
*   Directional Accuracy
*   Clear comparison across models and horizons

**Guide:**
*   **18–20:** Complete evaluation + clear tables + strong comparison
*   **14–17:** Minor gaps
*   **10–13:** Incomplete metrics or unclear comparison
*   **0–9:** Weak or incorrect evaluation

#### 4. Interpretation & Insights (25 pts) ⭐ (MOST IMPORTANT)
**What you are evaluating:**
*   Economic meaning of results
*   ML vs OLS discussion
*   Feature importance interpretation
*   Financial reasoning

**Guide:**
*   **22–25:** Deep insight, strong financial logic, clear explanations
*   **17–21:** Good interpretation, some depth missing
*   **12–16:** Mostly descriptive, limited insight
*   **0–11:** No meaningful interpretation

👉 **This is where you separate:**
*   “coders” vs “thinkers”

### 5. Writing, Presentation & Technical Quality (15 pts)
**What you are evaluating:**
*   Organization of paper
*   Clarity of writing
*   Clean tables/figures
*   Reproducible code (appendix)

**Guide:**
*   **13–15:** Clear, professional, well-structured
*   **10–12:** Minor clarity issues
*   **7–9:** Disorganized or unclear sections
*   **0–6:** Poor presentation

### ⭐ BONUS (UP TO +5 pts)
**Award for:**
*   Rolling window validation
*   Trading strategy
*   Hybrid FE + ML
*   Exceptional originality
