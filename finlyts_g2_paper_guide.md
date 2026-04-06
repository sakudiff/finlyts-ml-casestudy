## Guide to "Hybrid Econometric and Machine Learning Modeling of Philippine Stock Returns"

### 1. Project Overview & Objective

**What the paper is:** This research paper, titled "Hybrid Econometric and Machine Learning Modeling of Philippine Stock Returns," specifically focuses on a multi-horizon predictive analysis of Bank of the Philippine Islands (BPI) on the Philippine Stock Exchange (PSE).

**Core Objective:** The primary goal is to develop and compare both traditional econometric (OLS) and various machine learning (Random Forest, SVM, XGBoost, ANN) models to predict multi-horizon stock returns (1-day, 3-day cumulative, and 5-day cumulative) for a selected Philippine-listed company (BPI). The evaluation extends beyond mere statistical metrics to include economic significance and model interpretability, especially in a frontier market context.

**Big Picture:** This is a comprehensive attempt to assess the efficacy of advanced quantitative methods for equity return prediction in a less-efficient market, critically examining performance stability and economic relevance across different forecasting horizons and algorithmic complexities.

**Layman's Explanation:**
Imagine trying to guess if a stock will go up or down. This paper tries to do just that for a big bank in the Philippines (BPI), which is a key player in the economy (godoy2025measuring). We're comparing old-school math methods (OLS) with newer, smarter computer programs (Machine Learning like AI) to see which is better at predicting stock price movements for tomorrow, the next three days, and the next five days. It's not just about being right sometimes; it's about being consistently right and understanding *why* the predictions work (or don't).

### 2. Data and Feature Engineering

**What was done:**
*   **Asset Selection:** BPI.PS (target) and BDO.PS (competitor) were chosen due to their liquidity, sector relevance, and systemic importance within the Philippine banking sector, serving as proxies for domestic monetary policy transmission and macroeconomic health.
*   **Data Sources & Period:** Daily data from January 2, 2019, to April 1, 2026 (7.25 years, 1,853 observations) was collected.
    *   Local: BPI, BDO, PSEi, USD/PHP (Investing.com).
    *   Global Macro: U.S. 10-Year Treasury Yield (`^TNX`) (Yahoo Finance/API) – critical for capturing global liquidity conditions ("global gravity").
*   **Data Preprocessing:**
    *   Standardization of column headers, date parsing, numeric conversion, and scaling for volume data.
    *   Construction of a continuous daily calendar index, with Last-Observation-Carried-Forward (LOCF) imputation for missing values due to asynchronous market holidays.
    *   **CRITICAL:** Strict adherence to "no look-ahead bias" for all feature construction.
*   **Target Variables:** Three distinct log-return targets were constructed, each requiring separate model estimation:
    1.  1-day ahead return ($r_{t+1}$). 
    2.  3-day cumulative return ($r_{t:t+3}$).
    3.  5-day cumulative return ($r_{t:t+5}$).
*   **Feature Engineering (Categorical Breakdown):**
    *   **Trend Indicators:** MA(5, 10, 20), EMA(12, 26), MA5/MA20 ratio.
    *   **Momentum Indicators:** Lagged returns (1, 2, 3 days), Momentum (5, 10 days), RSI(14), MACD and signal line.
    *   **Volatility Indicators:** Rolling Standard Deviation (5, 10, 20 days), Volatility Ratio (short-term/long-term).
    *   **Market & Cross-Asset Features:** PSEi returns (current & lagged), BDO returns, USD/PHP returns, U.S. 10-Year Treasury Yield returns.
    *   **Feature Transformations:** Interaction term ($r_{t-1} 	imes 
sigma_{5,t}$), Squared volatility ($
sigma_{5,t}^2$).
    *   **Regime Variable:** `HiVol_t` (binary indicator for 20-day rolling std dev > 2x historical mean, calculated only on training data).
*   **Descriptive Statistics:** Key findings include significant excess kurtosis (BPI 1-day: 6.66, US 10Y: 45.00) and negative skewness (-0.28 for BPI 1-day returns), justifying non-linear models. The `HiVol_t` variable indicated stress periods, but only for 2.2% of observations, implying that models learning these rare events risk overfitting to noise.

**Why these findings/approaches:**
*   The data selection ensured representativeness and systemic importance for the target market.
*   The inclusion of specific global macroeconomic variables (US 10Y, USD/PHP) is theoretically motivated by their influence on capital flows and domestic liquidity in frontier markets.
*   Comprehensive feature engineering aims to capture various market dynamics (trend, momentum, volatility) and cross-asset relationships, as dictated by the assignment.
*   The "no look-ahead bias" rule is a non-negotiable principle in financial time series modeling, preventing spurious correlations and ensuring out-of-sample validity.
*   The observed non-Gaussian return distributions (fat tails, skewness) are characteristic of financial data, especially in emerging markets, making linear models inadequate for capturing tail risks and asymmetric dynamics.

**Layman's Explanation:**
We gathered a lot of daily information (over 7 years!) about BPI's stock, its competitor BDO, the overall Philippine stock market (PSEi), the US Dollar to Philippine Peso exchange rate, and US interest rates. These last two are super important because global money movements affect local markets a lot (tancangco2024effects). We also calculated many "indicators" from these numbers, like how fast prices are moving or how volatile they are.

**Crucially, we made sure not to "cheat"**: we only used information that would have been available *before* the day we're trying to predict. You can't use tomorrow's news to predict today's stock price!

We also noticed that stock prices don't behave nicely like a bell curve; they have "fat tails" (meaning extreme ups and downs happen more often than you'd expect) and are often "skewed" (more big drops than big rises, or vice versa). This is why simple math models often struggle, and we need smarter, non-linear tools (gu2020empirical).

### 3. Methodology

**What was done:**
*   **Data Partitioning:** A strict chronological 80/20 time-series split was used for training and testing, respectively. This is crucial for financial data to simulate real-world prediction and prevent look-ahead bias.
*   **Models Estimated:**
    1.  **OLS (Benchmark):** Linear regression with Newey-West standard errors for robust inference against heteroskedasticity and autocorrelation.
    2.  **Random Forest (RF):** Ensemble of 500 decision trees (`ranger`), tuned for `mtry` and `min_n`.
    3.  **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel (`kernlab`), with mandatory feature standardization (training set derived only), tuned for cost ($C$) and kernel width ($
sigma$).
    4.  **XGBoost:** Gradient boosting (`xgboost`), optimized with regularization, tuned for learning rate, tree depth, and L1/L2 penalties.
    5.  **Artificial Neural Network (ANN):** Single-layer perceptron (`nnet`), regularization via decay parameter, tuned for hidden units and decay rate.
*   **Validation & Hyperparameter Tuning:**
    *   Rolling origin cross-validation (`rsample::rolling_origin()`) to preserve temporal integrity.
    *   Latin Hypercube grid search for efficient exploration of parameter space.
*   **Evaluation Metrics:**
    *   **Statistical:** RMSE, MAE, R².
    *   **Financial (Primary):** Directional Accuracy (DA) – frequency of correct sign predictions.
    *   Comparison against a naïve lagged-return benchmark.
*   **Overfitting Assessment:** Explicitly quantified by comparing Training vs. Testing performance metrics.
*   **Ethical Reporting:** Emphasized open science, reproducible R code, and transparency.

**Why these findings/approaches:**
*   **Time-series split:** Absolutely fundamental for financial prediction, as random splitting would imply future information is available.
*   **Model Selection:** Covers a spectrum from simple linear (OLS) to complex non-linear models (ML) to thoroughly test the hypothesis of ML superiority.
*   **Hyperparameter Tuning:** Essential for optimizing ML models and avoiding suboptimal performance or overtuning.
*   **Directional Accuracy:** A critical metric in finance, as correctly predicting the *sign* of a return is often more valuable for trading strategies than predicting its exact magnitude.
*   **Overfitting Diagnosis:** Recognizing and quantifying overfitting is paramount in financial machine learning, where signal-to-noise ratios are notoriously low. Without it, models appear performant in-sample but fail catastrophically out-of-sample.

**Layman's Explanation:**
To test our predictions fairly, we split our historical data. We trained our models on the first 80% of the data, pretending that was "the past." Then, we let them make predictions on the remaining 20%, which was "the future" they hadn't seen yet. This is like teaching a student with old exam papers and then giving them a new, unseen exam.

We used a variety of prediction tools:
*   **OLS:** The simplest, like drawing a straight line through data.
*   **Random Forest:** Like having 500 different "decision trees" (simple flowcharts) vote on the outcome.
*   **SVM:** Finds the best "boundary" to separate different outcomes, even in complex data.
*   **XGBoost:** Builds many simple "trees" one after another, each one trying to fix the mistakes of the previous one.
*   **ANN:** A basic "neural network," inspired by how brains work, looking for patterns.

We "tuned" these models to make sure they were performing their best, and we specifically watched out for "overfitting." Overfitting is when a model learns the past data *too well*, including all its quirks and random noise, so it fails when it sees new data. It's like a student who memorizes answers instead of understanding the concepts (bustos2025machine).

Our main goal was to see if the models could predict the *direction* of the stock movement (up or down), which we call "Directional Accuracy." This is often more useful for investing than guessing the exact price.

### 4. Results

**What our findings are (in depth):**

*   **OLS (Baseline) Performance:**
    *   **DA:** Improves monotonically with horizon: 1-day (33.1%), 3-day (49.0%), 5-day (51.1%). Significantly outperforms the naive benchmark directionally.
    *   **R²:** Negative out-of-sample R² across all horizons, meaning it explains less variance than a simple mean prediction. Captures directional drift but not magnitude.
    *   **Overfitting:** Minimal; Train RMSE/DA very close to Test RMSE/DA. Establishes a stable, if directionally focused, linear benchmark.
*   **Machine Learning Model Performance (Overview):**
    *   **Random Forest (RF):** Exhibited severe overfitting. Training DA (83.2% at 1-day, 91.3% at 5-day) collapsed to Test DA (43.7% at 1-day, 47.2% at 5-day). Test R² was negative (-0.134 at 5-day), indicating worse than mean prediction. Worst Test RMSE at 5-day.
    *   **XGBoost:** Moderate overfitting. Achieved a marginally positive R² (0.008 at 1-day) but was economically negligible. 1-day Test DA (40.4%) was *below* the naive benchmark (42.9%). Recovers for 3/5-day but still below OLS DA.
    *   **Support Vector Machine (SVM):** Most robust generalization. Achieved the highest out-of-sample DA (49.0% at 5-day). Narrowest Train-Test RMSE gaps across all horizons.
    *   **Artificial Neural Network (ANN):** Closely mirrored SVM performance (49.2% DA at 5-day), also showing narrow Train-Test RMSE gaps.
*   **Overfitting Assessment (Quantitative):**
    *   RF's RMSE gap (Train-Test) widened from -0.0093 (1-day) to -0.0205 (5-day).
    *   SVM's gap remained nearly flat (-0.0002 to -0.0005).
    *   Hierarchy of overfitting severity: RF $\gg$ XGB $>$ OLS $>$ SVM $\approx$ ANN, correlating with model complexity and effective degrees of freedom.
*   **Horizon Decay:** All models showed monotonically increasing RMSE with longer horizons. SVM had the shallowest RMSE slope, indicating graceful degradation. The naive benchmark's DA peaked at 3-day, suggesting a characteristic autocorrelation timescale.
*   **Naive Benchmark Comparison:** 14 out of 15 model-horizon combinations beat the naive benchmark directionally. XGBoost at 1-day was the sole exception. Largest margins belonged to OLS at multi-day horizons.
*   **Variable Importance & Coefficients:**
    *   **OLS Coefficients:**
        *   1-day: Dominated by `lag1` and `r_bpi` (negative coefficients, confirming short-horizon mean reversion). `lag_psei` also significant.
        *   3-day: Mean reversion persists across `lag1`, `lag2`, `lag3`. `r_fx` (negative, peso appreciation predicts positive BPI returns) and the `interact` term (lagged returns amplified in high vol) become significant. `vol5_sq` also shows a negative convex relationship.
        *   5-day: `r_fx` becomes highly significant and almost doubles in magnitude, highlighting currency channel dominance. `vol5` (volatility risk premium), `vol_ratio`, `vol5_sq` intensify. `ma_ratio` and `lag_psei` gain explanatory power.
    *   **RF Importance:** 1-day: `lag1`, `r_us10y`. 5-day: `ma_ratio`, `mom5`, `r_fx`. Indicates a shift from short-term autoregression/global liquidity to structural trend signals.
    *   **XGBoost Importance:** Higher sensitivity to `r_bpi`. `interact` more prominent. `hi_vol`, `vol5_sq` gain importance at 5-day. `r_us10y` ranks lower than in RF, suggesting a focus on domestic dynamics.
*   **Forecasting Profile Diagnostics (Scatter Plots):**
    *   All models cluster predictions tightly around zero, reflecting low signal-to-noise.
    *   **RF:** High dispersion, "striped" pattern (step-function nature), overconfident extrapolation. Visual evidence of overfitting.
    *   **XGBoost:** Tighter envelope, conservative bias (underestimates large moves) due to regularization.
    *   **SVM:** Most robust, conservative clustering around the mean at 1-day (Bayesian response). Clearer positive slope at 5-day, identifies directional sign accurately.
    *   **ANN:** Mirrors SVM, cautious forecast, but with slightly wider prediction spread.

**Layman's Explanation:**

Here's what we found when our models tried to predict the "future" (the 20% of data they hadn't seen):

*   **The Simple Model (OLS):** Surprisingly good at guessing the *direction* of the stock (up or down), especially for 3 and 5-day predictions, even beating simpler guesses. It didn't "overfit" (memorize the past too well) and was very stable. However, it couldn't tell us *how much* the stock would move, just the general direction.
*   **The "Smart but Naive" Model (Random Forest):** This one was a superstar on the past data, seemingly predicting almost everything perfectly. But on new, unseen data, it completely fell apart. Its predictions were often worse than just guessing randomly! This is the classic example of **overfitting**: it memorized the training data's noise instead of learning general rules.
*   **The "Smarter, More Careful" Models (SVM and ANN):** These were the quiet winners. They weren't flashy on the old data, but they were the most reliable and accurate on new data, especially for 5-day predictions. They managed to avoid overfitting because they have built-in "caution" or "regularization" that stops them from getting obsessed with every tiny detail of the past (gu2020empirical).
*   **XGBoost:** Better than Random Forest, but still struggled a bit, especially for next-day predictions. It was okay, but not as consistently good as SVM or ANN.
*   **The "Longer the Forecast, the Harder it Gets" Rule:** For almost all models, predicting further into the future (5 days vs. 1 day) generally meant more errors. It's harder to guess what happens next week than tomorrow. But, interestingly, the 5-day predictions for SVM and ANN were still quite good at guessing direction.
*   **What Drives Predictions?**
    *   **For short predictions (1-day):** It's mostly about what the stock did yesterday or the day before. If it jumped very high, it might "revert" a bit the next day. Also, global interest rates played a role.
    *   **For longer predictions (3 and 5-day):** The value of the Philippine Peso against the US Dollar became super important. If the Peso strengthened, BPI's stock tended to rise. Also, overall market trends and how volatile the stock was played a bigger role.

In short, the really fancy, complex models (like Random Forest) often failed because they tried *too hard* to learn every wobble in the past. The models that were smart but also "careful" (SVM, ANN) were the most reliable.

### 5. Discussion (Why these findings and their implications)

**Major Points:**
*   **ML Overfitting vs. Regularization:** The study directly challenges the "more parameters equals better predictions" adage in finance. High-capacity models like Random Forest overfit severely, memorizing noise from past regimes (e.g., COVID-19, 2022 inflation spike) that do not generalize. Conversely, explicitly regularized models (SVM, ANN) demonstrate robust out-of-sample performance, achieving superior directional accuracy with minimal overfitting. This is a critical lesson for deploying ML in low-signal-to-noise environments.
*   **OLS Competitiveness & Interpretability:** Despite its linear limitations, OLS remains competitive in directional accuracy, particularly at longer horizons. Its transparent, interpretable coefficients make it a compelling choice for practitioners who require explainable predictions (e.g., for regulatory compliance or risk management). This highlights the trade-off between predictive power and interpretability.
*   **Multi-Horizon Dynamics & Signal Decay:**
    *   **1-day:** Dominated by microstructure noise, making directional prediction challenging for all models.
    *   **3-day:** Shows significant improvement in DA, suggesting a smoothing effect that exposes structural macroeconomic momentum and absorbs scheduled information releases.
    *   **5-day:** Emerges as the "sweet spot" for directional predictability, where regularized models (SVM, ANN) and OLS achieve their highest DA. This implies that persistent trends and macro signals are more reliably captured over a weekly horizon.
    *   The naive benchmark's DA peaking at 3-day suggests a characteristic autocorrelation timescale for BPI returns, beyond which individual lagged returns lose predictive power.
*   **Economic Interpretation of Features:**
    *   **Global Gravity:** The U.S. 10-Year Treasury Yield strongly influences short-horizon PSE returns, confirming that global liquidity conditions drive capital flows to frontier markets.
    *   **Mean Reversion:** Dominant at short horizons (1-day), reflecting the swift correction of order flow imbalances in a relatively thin market.
    *   **Currency Channel:** USD/PHP exchange rate significantly impacts BPI returns at longer horizons (3- and 5-day), via its effects on bank profitability, capital inflows, and monetary policy.
    *   **Volatility Risk Premium:** Elevated short-term volatility predicts positive cumulative returns over a weekly window, indicating compensation for bearing uncertainty.
    *   **Adaptive Market Hypothesis (AMH) Confirmation:** The performance divergence between models, particularly the regime-specific failure of unregularized ML, strongly supports the AMH. The market is an evolving ecosystem, and predictability is a time-varying function of changing environmental conditions, demanding adaptive and robust models.

**Minor Points:**
*   The `HiVol` variable, though impactful, flagged very few observations, making it prone to overfitting by high-capacity models.
*   XGBoost's performance, while better than RF, still showed a conservative bias and struggled with out-of-sample R², hinting that its regularization was insufficient or its architecture less suited for this specific market.
*   ANN's slightly wider prediction spread compared to SVM suggests a subtle vulnerability to overconfident extrapolation.

**Practical Implications:**
*   **Model Selection:** For weekly directional positioning, SVM or ANN are preferred due to their robustness. For highly interpretable, though less performant, predictions, OLS is suitable. Random Forest should be avoided in production for this task.
*   **Multi-Horizon Strategy:** Instead of a single forecast, a "consensus signal" from models predicting across multiple horizons is recommended for stronger, more reliable predictions.
*   **Focus on Direction:** The consistent finding of reasonable Directional Accuracy, even with negative R² for many models, implies that successful trading in this market might hinge on correctly identifying the *direction* of movement rather than precise magnitude.

**Layman's Explanation:**

Here's what all these results mean:

*   **Fancy Doesn't Always Mean Better:** Just because an AI model is complex and uses lots of data doesn't mean it's good. In fact, the most complex one (Random Forest) was the worst because it got "distracted" by random events in the past and couldn't apply its "learning" to new situations. The simpler, more cautious AI models (SVM, ANN) were the winners.
*   **The Old Way Still Has Game:** The very basic math model (OLS) performed surprisingly well, especially for longer predictions. And a huge bonus for OLS: you can easily see *why* it's making a prediction. This "transparency" is super important for investors and regulators who need to understand the logic.
*   **It's Hard to Predict Tomorrow, Easier Next Week:** Trying to guess what happens in the stock market *tomorrow* is incredibly noisy, like trying to predict exactly where a single raindrop will land. But if you look at a 3-day or 5-day window, the bigger economic trends and influences start to show through, making predictions a bit easier. The 5-day window seemed to be the sweet spot.
*   **What Really Matters to Philippine Stocks?**
    *   **Global Interest Rates:** How high US interest rates are makes a big difference, often pushing money out of developing markets like the Philippines.
    *   **The Peso's Strength:** A stronger Philippine Peso (meaning the USD/PHP exchange rate goes down) tends to be good for BPI's stock over a few days. This is because it affects bank profits, attracts more foreign money, and can lead to easier money policies by the central bank (tancangco2024effects).
    *   **Stock's Past Behavior:** Sometimes, if a stock moves very far in one direction, it tends to "snap back" a bit the next day. Other times, if it's been trending, that trend continues for a few days.
*   **The Market Changes:** The models that adapted best were those that didn't just memorize patterns from chaotic times (like the COVID pandemic) but learned more general rules that held true even when the market environment became calmer. This idea, that markets are always changing and adapting, is called the "Adaptive Market Hypothesis" (ahmed2022adaptive; huynh2025composite).
*   **What Should Investors Do?** Don't trust overly complex AI models that claim to predict everything perfectly. The more "cautious" AI models (SVM, ANN) are likely your best bet for general direction over a week. And always remember, simply knowing *which way* the stock will go (up or down) is a huge advantage, even if you don't know the exact amount (krauss2017deep).

### 6. Conclusion and Recommendations

**Economic Significance:**
The findings are indeed financially meaningful and strongly align with the Adaptive Market Hypothesis (AMH) rather than the Efficient Market Hypothesis (EMH). While daily equity return prediction remains challenging due to low signal-to-noise, consistent directional accuracy approaching 50% at a 5-day horizon (or exceeding it with OLS) is economically significant. Such an edge, even if small, can generate alpha when combined with proper risk management. The study demonstrates that predictability is not constant but contingent on market regimes and model robustness.

**Practical Implications:**
PSE analysts and practitioners should prioritize models with explicit regularization (SVM, ANN) for multi-horizon return forecasts, particularly for weekly predictions. The OLS model, despite its simplicity, offers valuable interpretability and competitive directional accuracy, making it suitable for scenarios demanding transparency. High-capacity, unregularized machine learning models (like Random Forest in this case) are prone to severe overfitting in this market context and should be deployed with extreme caution, if at all. The insight that a multi-horizon consensus signal improves reliability is also a key takeaway.

**Limitations and Future Research:**
*   **Additional Macro Variables:** Future work could incorporate a broader set of global and domestic macroeconomic indicators, potentially including sentiment data, to enrich the feature space and capture more nuanced economic drivers.
*   **High-Frequency Data:** Investigating higher-frequency data (e.g., intraday) could reveal different microstructure dynamics and potential for short-term statistical arbitrage, though it would also introduce greater noise.
*   **Alternative Architectures:** Exploring more advanced deep learning architectures (e.g., LSTMs with attention mechanisms) specifically designed for sequential data, coupled with rigorous regularization techniques, could yield further improvements.
*   **Trading Strategy Simulation:** Explicitly simulating a trading strategy based on the best-performing models, incorporating realistic transaction costs and slippage, would provide a more concrete measure of economic utility.
*   **Dynamic Weighting:** Research into dynamically weighting models or features based on real-time regime detection could enhance adaptive capabilities.

**Layman's Explanation:**

**So, what's the big takeaway for investors?**
Even in a tricky market like the Philippines, you *can* find ways to predict if a stock like BPI will go up or down. It's not a perfect crystal ball, but having a model that's right almost 50% of the time over a week is a significant advantage if you manage your risks well (bustos2025machine).

**What kind of prediction tools should you use?**
Stick with the "careful" AI models like SVM or ANN for longer-term predictions (like a week out). If you need to understand *why* a prediction is being made, the simpler OLS model is very useful and surprisingly effective. Avoid the overly complex AI models (like Random Forest) that try to be perfect; they usually fail in the real world.

**What's next for this research?**
We could look at even more global economic factors, or dig into how stocks move minute-by-minute. We could also try even newer, fancier AI models, but always remembering that "caution" (regularization) is key. The ultimate goal is to see if these predictions can actually make money in a real trading scenario.
