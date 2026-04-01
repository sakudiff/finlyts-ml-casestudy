# **Hybrid Machine Learning and Econometric Modeling for Philippine Stock Returns: Market Efficiency, Macroeconomic Drivers, and Multi-Horizon Predictability**

## **1\. Introduction to Asset Pricing in Frontier Markets**

The intersection of financial econometrics and advanced computational modeling has fundamentally altered the landscape of empirical asset pricing. Historically, the pursuit of modeling expected equity returns has relied on linear regression frameworks that prioritize unbiased parameter estimation and strict inferential assumptions. However, financial markets—particularly emerging and frontier ecosystems like the Philippine Stock Exchange (PSE)—exhibit highly complex, non-stationary, and non-linear behaviors that frequently violate the foundational assumptions of classical statistical models. As global liquidity dynamics, macroeconomic volatility, and idiosyncratic regulatory shifts interact, the resulting asset price movements demand a more robust, adaptive, and high-dimensional analytical approach.

The transition from classical linear modeling to machine learning architectures represents a paradigm shift from prioritizing in-sample explanatory power to maximizing out-of-sample predictive accuracy. This exhaustive research report delivers a highly detailed examination of hybrid modeling techniques applied to stock return predictability, with a specific focus on the Philippine equity market. By analyzing the theoretical friction between econometrics and machine learning, this discourse evaluates how models such as Random Forests, Extreme Gradient Boosting (XGBoost), Support Vector Machines (SVM), and Artificial Neural Networks (ANN) isolate predictive signals amidst overwhelming market noise.1

Furthermore, this analysis investigates the philosophical implications of these predictive models on established financial theories. High-frequency predictability directly challenges the Weak-Form Efficient Market Hypothesis (EMH), suggesting that the Adaptive Market Hypothesis (AMH) provides a more precise framework for understanding frontier markets. Within the Philippine context, this theoretical lens is applied to the dominant banking sector—specifically examining the heavyweights BDO Unibank (BDO) and the Bank of the Philippine Islands (BPI)—to dissect how macroeconomic indicators, particularly the USD/PHP exchange rate and remittance-driven liquidity, act as deterministic features in equity valuation. Finally, the report explores the temporal dimension of predictive modeling, detailing the mathematical and empirical phenomena of signal decay and temporal aggregation across multi-horizon forecasts.

## **2\. Bibliographic Foundations and Source Mandates**

A robust theoretical foundation is required to contextualize the comparative advantages of machine learning in asset pricing. The research architecture of this report is built upon a specific corpus of peer-reviewed literature and institutional working papers. Per the structural mandates of this inquiry, the foundational texts—including the core seed citations—are integrated directly into this section. These bibliographic anchors provide the empirical baseline for evaluating algorithmic performance against traditional econometric benchmarks across the Philippine and global equity markets.

Code snippet

@article{gu2020empirical,  
  author \= {Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},  
  title \= {Empirical Asset Pricing via Machine Learning},  
  journal \= {The Review of Financial Studies},  
  volume \= {33},  
  number \= {5},  
  pages \= {2223--2273},  
  year \= {2020},  
  doi \= {10.1093/rfs/hhaa009}  
}  
% SUMMARY: This seminal study performs a comparative analysis of machine learning methods for the canonical problem of empirical asset pricing, establishing that neural networks and tree-based ensembles substantially outperform traditional linear regressions. The researchers trace the predictive gains to the models' capacity to capture complex nonlinear predictor interactions involving momentum, liquidity, and volatility. Ultimately, the findings provide a new benchmark for out-of-sample predictability in asset pricing and demonstrate significant economic gains for investors.

Code snippet

@article{krauss2017deep,  
  author \= {Krauss, Christopher and Do, Xuan Anh and Huck, Nicolas},  
  title \= {Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S\\\&P 500},  
  journal \= {European Journal of Operational Research},  
  volume \= {259},  
  number \= {2},  
  pages \= {689--702},  
  year \= {2017},  
  doi \= {10.1016/j.ejor.2016.10.031}  
}  
% SUMMARY: This research implements and evaluates the effectiveness of deep neural networks, gradient-boosted trees, and random forests for daily statistical arbitrage on the S\&P 500\. The methodology involves generating one-day-ahead trading signals based on probability forecasts of outperformance, finding that a simple, equal-weighted ensemble of these models produces daily out-of-sample returns exceeding 0.45 percent. The empirical results mount a severe challenge to the semi-strong form of market efficiency, although profitability has exhibited a declining trend in recent years.

Code snippet

@article{fischer2018deep,  
  author \= {Fischer, Thomas and Krauss, Christopher},  
  title \= {Deep learning with long short-term memory networks for financial market predictions},  
  journal \= {European Journal of Operational Research},  
  volume \= {270},  
  number \= {2},  
  pages \= {654--669},  
  year \= {2018},  
  doi \= {10.1016/j.ejor.2017.11.054}  
}  
% SUMMARY: The authors deploy Long Short-Term Memory (LSTM) networks to predict out-of-sample directional movements for S\&P 500 constituent stocks from 1992 to 2015\. The study demonstrates that LSTMs outperform memory-free classification methods such as random forests and standard deep neural networks by successfully extracting complex sequential dependencies. Furthermore, the model isolates a distinct pattern of high volatility and short-term reversal, allowing for the formulation of a highly profitable rules-based trading strategy.

Code snippet

@techreport{godoy2025measuring,  
  author \= {Godoy, Jorjin F.},  
  title \= {Measuring the Systemic Risk Contribution of Philippine Industries and Conglomerate Groups},  
  institution \= {Bangko Sentral ng Pilipinas},  
  type \= {Discussion Paper Series},  
  number \= {2025-08},  
  year \= {2025},  
  doi \= {10.20955/r.bsp.2025.08}  
}  
% SUMMARY: This working paper assesses the systemic risk contribution (SRISK) of various Philippine sectors, finding that SRISK levels during the COVID-19 pandemic tripled compared to the Global Financial Crisis. The methodology employs a Vector Autoregression (VAR) framework to reveal that macroeconomic shocks to GDP, policy rates, and the USD/PHP exchange rate significantly drive systemic vulnerability. The banking sector is identified as the dominant contributor to systemic risk, highlighting the critical interconnection between macroeconomic stability and financial equity valuations.

Code snippet

@article{huynh2025composite,  
  author \= {Huynh, Tran Trong and Dinh, Thi Thu Hong},  
  title \= {A Composite Efficiency Index for ASEAN Foreign Exchange Markets},  
  journal \= {International Journal of Analysis and Applications},  
  volume \= {23},  
  number \= {1},  
  pages \= {298},  
  year \= {2025},  
  doi \= {10.28924/2291-8639-23-2025-298}  
}  
% SUMMARY: This study examines the dynamics of foreign exchange market efficiency in six ASEAN economies, including the Philippines, over a twenty-five-year period. Utilizing a Composite Efficiency Index (CEI) derived via principal component analysis, the authors demonstrate that market inefficiency has generally declined over time but spikes dramatically during systemic crises. The findings strongly support the Adaptive Market Hypothesis, showing that efficiency is multidimensional, event-driven, and highly variable across different economic regimes.

Code snippet

@article{lim2026dynamic,  
  author \= {Lim, Brian Godwin and Dayta, Dominic and Tiu, Benedict Ryan and Tan, Renzo Roel and Garces, Len Patrick Dominic and Ikeda, Kazushi},  
  title \= {Dynamic factor analysis of price movements in the Philippine Stock Exchange},  
  journal \= {Financial Innovation},  
  volume \= {12},  
  number \= {11},  
  pages \= {1--25},  
  year \= {2026},  
  doi \= {10.1186/s40854-025-00807-7}  
}  
% SUMMARY: This empirical paper utilizes dynamic factor modeling and maximum likelihood estimation to analyze the latent drivers of price movements within the Philippine Stock Exchange (PSE). The researchers validate their multi-factor extraction against traditional capital asset pricing models, revealing that specific unobserved common factors distinctly represent market trends and volatility. The study concludes that these dynamic factors can significantly reduce out-of-sample prediction errors, emphasizing the value of complex econometric frameworks in frontier markets.

Code snippet

@article{ahmed2022adaptive,  
  author \= {Ahmed, Haydory Akbar},  
  title \= {Does the Adaptive Market Hypothesis Reconcile Behavioral Finance and the Efficient Market Hypothesis?},  
  journal \= {Risks},  
  volume \= {10},  
  number \= {9},  
  pages \= {168},  
  year \= {2022},  
  doi \= {10.3390/risks10090168}  
}  
% SUMMARY: This study critically evaluates the theoretical bridge between classical market efficiency and behavioral finance provided by the Adaptive Market Hypothesis (AMH). The author conducts extensive empirical testing across multiple international markets, confirming that return predictability is a cyclical phenomenon driven by shifting macroeconomic and behavioral regimes. The findings argue that investors must actively manage portfolios to capitalize on the transient inefficiencies dictated by the AMH.

Code snippet

@article{waldow2021machine,  
  author \= {Waldow, Fabian and Schnaubelt, Matthias and Krauss, Christopher and Fischer, Thomas G.},  
  title \= {Machine Learning in Futures Markets},  
  journal \= {Journal of Risk and Financial Management},  
  volume \= {14},  
  number \= {3},  
  pages \= {119},  
  year \= {2021},  
  doi \= {10.3390/jrfm14030119}  
}  
% SUMMARY: This research investigates the application of various machine learning algorithms, including tree-based ensembles and neural networks, to predict short-term directional movements in global futures markets. The empirical results demonstrate that short-term predictive features consistently exhibit the largest explanatory power, primarily driven by localized mean-reversion and short-term reversal effects. The study confirms that hybrid machine learning models generate significant alpha over classical buy-and-hold strategies in high-frequency trading environments.

Code snippet

@article{bustos2025machine,  
  author \= {Bustos, O. and Pomares-Quimbaya, A. and Stellian, R.},  
  title \= {Machine learning, stock market forecasting, and market efficiency: A comparative study},  
  journal \= {International Journal of Data Science and Analytics},  
  volume \= {20},  
  number \= {4},  
  pages \= {6815--6839},  
  year \= {2025},  
  doi \= {10.1007/s41060-025-00815-x}  
}  
% SUMMARY: This comprehensive comparative study analyzes the performance of diverse machine learning architectures in forecasting equity returns across multiple market efficiency regimes. The authors highlight the persistent gap between statistical predictability metrics (such as R-squared) and actual economic viability when real-world transaction costs are applied. The study concludes that while deep learning models effectively capture non-linear patterns, the translation of these patterns into profitable strategies remains highly sensitive to market friction.

Code snippet

@techreport{tancangco2024effects,  
  author \= {Tancangco, Jose Adlai M. and Parcon-Santos, Hazel C.},  
  title \= {The Effects of Exchange Rates and Inflation Targeting on Exports in the RCEP Region},  
  institution \= {Bangko Sentral ng Pilipinas},  
  type \= {Discussion Paper Series},  
  number \= {2024-22},  
  year \= {2024},  
  doi \= {10.20955/r.bsp.2024.22}  
}  
% SUMMARY: This central bank working paper explores the macroeconomic impacts of exchange rate volatility and inflation-targeting regimes on cross-border trade and economic liquidity. The authors demonstrate that domestic currency depreciation substantially enhances export competitiveness, thereby injecting liquidity into the domestic economy. The findings provide critical context for understanding how central bank monetary policy and exchange rate management dynamically alter the fundamental valuation of domestic equities.

The preceding literature collectively establishes that financial time series forecasting is no longer bound by the restrictive assumptions of linear parsimony. Instead, the application of highly parameterized architectures offers measurable economic utility, fundamentally reshaping how market practitioners extract alpha from complex data structures.1

## **3\. The Machine Learning vs. Econometrics Debate: Empirical Justifications**

The mathematical formalization of asset pricing has historically relied heavily on the Ordinary Least Squares (OLS) estimator. In traditional econometric theory, OLS is celebrated for satisfying the Gauss-Markov assumptions, thereby providing the Best Linear Unbiased Estimator (BLUE). The classical econometrician optimizes for the minimization of squared residuals to identify the fundamental, causal drivers of expected returns. While classical econometrics excels in parameter interpretability and strict inferential diagnostics, it systematically fails when faced with the high-dimensional, multicollinear, and non-stationary realities of modern equity markets.

### **3.1 The Mathematical Limitations of Linear Parsimony**

The standard econometric approach models the excess return of an asset, ![][image1], as a linear combination of predictive features ![][image2]:

![][image3]  
In modern financial environments, the number of potential predictors (![][image4])—ranging from fundamental accounting ratios to high-frequency technical indicators—often approaches or exceeds the number of temporal observations (![][image5]). In such high-dimensional spaces, the OLS estimator ![][image6] becomes unstable or mathematically non-invertible due to severe multicollinearity, leading to an explosion in estimator variance.1 This is the quintessential "curse of dimensionality." Even when regularization techniques such as Ridge (L2 penalty) or Lasso (L1 penalty) are applied to shrink coefficients and perform automated feature selection, the underlying functional form remains strictly additive and linear.5

As rigorously established by Gu, Kelly, and Xiu (2020), the true data generating process in equity markets is not linear. It involves deep, non-linear interactions between firm-level characteristics (e.g., momentum, trading liquidity, historical volatility) and broad macroeconomic state variables.1 An OLS model entirely misses these cross-sectional and temporal interactions unless an econometrician manually engineers specific interaction terms—a process that inevitably introduces severe human bias and data dredging into the modeling pipeline.

### **3.2 The Superiority of Tree-Based Ensembles**

To overcome the structural limitations of linear parsimony, tree-based ensemble methods—specifically Random Forests (RF) and Extreme Gradient Boosting (XGBoost)—have emerged as dominant frameworks for modeling cross-sectional stock returns.1 These algorithms reject the assumption of a global functional form. Instead, they partition the multidimensional predictor space into distinct hyper-rectangles using recursive binary splits, making them inherently adept at capturing non-linear thresholds and complex interaction effects without prior human specification.

Random Forests utilize bagging (bootstrap aggregating) to train deep, unpruned decision trees on random subsets of data and random subspaces of features. By averaging the predictions of thousands of uncorrelated trees, the algorithm drastically reduces model variance without inflating bias, making it highly robust to the extreme outliers typical of financial time series.6

Conversely, XGBoost employs a sequential learning architecture known as boosting. Each new, relatively shallow tree is specifically trained to optimize the residual errors of the previous ensemble. Mathematically, XGBoost achieves this by applying a second-order Taylor expansion to approximate the loss function, seamlessly integrating L1 and L2 regularization terms directly into the objective function to prevent overfitting.9 In the comprehensive comparative analysis conducted by Gu et al. (2020), both Random Forests and Gradient Boosted Trees demonstrated substantial out-of-sample predictive gains, frequently doubling the performance metrics (such as out-of-sample ![][image7] and portfolio Sharpe ratios) of the leading regression-based strategies documented in prior literature.1 The models consistently converged on a specific hierarchy of dominant predictive signals, prioritizing short-term momentum, trading liquidity, and return volatility over traditional fundamental accounting ratios.12

### **3.3 Deep Learning and Sequential Feature Abstraction**

While tree-based models excel at processing cross-sectional tabular data, Artificial Neural Networks (ANNs) and deep learning architectures are uniquely suited for dynamic feature abstraction and sequence learning. A standard feedforward ANN passes inputs through multiple hidden layers characterized by non-linear activation functions (such as ReLU or Sigmoid), mathematically allowing the network to approximate any continuous function given sufficient parameterization.9

However, for chronological financial time series, the Long Short-Term Memory (LSTM) network represents a critical architectural evolution. Fischer and Krauss (2018) demonstrated that LSTMs decisively outperform memory-free classification models—including standard deep neural networks, logistic regression, and random forests—when predicting daily out-of-sample directional movements for S\&P 500 constituents.3 LSTMs mitigate the vanishing gradient problem inherent in traditional Recurrent Neural Networks (RNNs) through a highly complex internal architecture of information gates:

![][image8]  
![][image9]  
These mathematically governed gates allow the network to explicitly "remember" long-term structural dependencies and actively "forget" transient market noise.15 The empirical deployment of LSTM models uncovered that optimal trading signals are often generated by identifying stocks exhibiting high localized volatility coupled with a short-term reversal return profile.14

| Model Classification | Primary Mechanism | Strengths in Financial Modeling | Structural Vulnerability |
| :---- | :---- | :---- | :---- |
| **OLS / Penalized Regression** | Residual variance minimization | High interpretability, explicit causal inference | Strict linear constraint, severe multicollinearity sensitivity |
| **Random Forest** | Bootstrap aggregating (Bagging) | Massive variance reduction, robustness to outliers | Complete inability to extrapolate beyond historical training bounds |
| **XGBoost / GBT** | Sequential residual optimization | Superior capture of deep non-linear interactions | Highly prone to overfitting on low signal-to-noise datasets |
| **LSTM Network** | Gated chronological sequence memory | Extraction of temporal dependencies and trend reversals | High computational cost, extreme hyperparameter sensitivity |

### **3.4 The "No Free Lunch" Theorem in Financial Forecasting**

Despite the immense empirical triumphs of machine learning, the foundational "No Free Lunch" (NFL) theorem of optimization dictates that no single algorithmic strategy will universally outperform all others across every possible problem space.17 In the context of asset pricing, this mathematical reality implies that the efficacy of a model is entirely conditional on the specific distributional properties of the target market. A highly parameterized neural network that perfectly captures the dense microstructure of the highly liquid S\&P 500 may fail catastrophically when applied to the structurally distinct, liquidity-constrained environment of the Philippine Stock Exchange.20

Consequently, the most robust approach to financial forecasting is the deployment of diverse hybrid ensembles. Krauss, Do, and Huck (2017) empirically validated this theory by demonstrating that an equal-weighted ensemble consisting of a deep neural network, a gradient-boosted tree, and a random forest achieved statistical arbitrage returns exceeding 0.45 percent per day prior to transaction costs.2 This ensemble approach effectively hedges against the unique algorithmic blind spots of individual models, leveraging the constraints of the NFL theorem to construct a composite hybrid system that continuously adapts to shifting market distributions and volatility regimes.21

## **4\. The Adaptive Market Hypothesis and Frontier Markets (PSE Focus)**

The demonstrable success of machine learning algorithms in extracting profitable, out-of-sample trading signals fundamentally challenges the bedrock of classical financial theory. Under Eugene Fama's (1970) Weak-Form Efficient Market Hypothesis (EMH), future asset prices cannot be predicted using historical price and volume data, as the market instantaneously and rationally prices in all historical information. If weak-form efficiency holds unconditionally, the sophisticated pattern recognition capabilities of LSTMs and XGBoost should theoretically yield an expected out-of-sample alpha of exactly zero.

### **4.1 Theoretical Departure from Weak-Form Efficiency**

Empirical evidence generated by modern machine learning firmly rejects unconditional weak-form efficiency in favor of a more dynamic and nuanced framework. Andrew Lo (2004) proposed the Adaptive Market Hypothesis (AMH), which elegantly applies the principles of evolutionary biology and behavioral ecology to financial ecosystems.22 Under the AMH, market efficiency is not an absolute, binary state but a fluid, continuous characteristic that fluctuates based on the macroeconomic environment. Market participants—driven by bounded rationality, behavioral biases, and shifting risk appetites—continuously adapt to new information, regulatory shifts, and exogenous macro shocks.25

When market environments are highly stable, fierce competition drives the ecosystem toward efficiency. However, during periods of structural stress, sudden liquidity dry-ups, or severe geopolitical shocks, evolutionary adaptation lags. This lag results in the temporary emergence of predictable, exploitable inefficiencies.27 The success of machine learning models is contingent precisely on their computational ability to detect and exploit these fleeting pockets of behavioral inefficiency faster than competing human market participants.

### **4.2 Mean Reversion and Momentum in the Philippine Stock Exchange**

The Philippine Stock Exchange (PSE) serves as an optimal, structurally distinct laboratory for evaluating the AMH. As a frontier-to-emerging market, the PSE exhibits unique microstructural frictions: highly concentrated market capitalization, lower aggregate trading volumes relative to developed global peers, and heightened vulnerability to the sudden reversal of international capital flows.29

Classical econometric analyses utilizing variance ratio tests often suggest that the PSE exhibits periods of overall informational efficiency.30 However, deeper non-linear probing via machine learning and dynamic factor analysis reveals profound deviations from the random walk hypothesis. Research targeting specific sectoral indices within the PSE confirms that while aggregate returns may appear unpredictable at the macroeconomic index level, distinct patterns of mean reversion and momentum manifest dynamically across specific forecasting horizons and volatility regimes.31

The existence of momentum effects in the PSE—where equities exhibiting high recent returns continue to persistently outperform—violates weak-form efficiency. In frontier markets, momentum is frequently driven by investor herding behavior, delayed institutional information processing, and the slow diffusion of global macroeconomic news into local asset prices.34 Conversely, mean reversion acts as an aggressive counter-force, typically triggered by retail overreactions to negative news sentiment or sudden liquidity shocks that force prices away from fundamental equilibrium. Hybrid machine learning models are particularly devastating in the PSE because they can dynamically toggle their internal weights between momentum-following and mean-reversion strategies depending on the prevailing volatility regime, perfectly encapsulating the behavioral adaptations described by the AMH.24

### **4.3 Evolutionary Finance in the ASEAN-5 Ecosystem**

The localized dynamics of the PSE are inextricably linked to broader systemic trends within the ASEAN-5 region. Empirical investigations into the foreign exchange and equity markets of Southeast Asia strongly corroborate the AMH. These studies reveal that inefficiency spikes predictably during systemic crises (such as the 2008 Global Financial Crisis or the COVID-19 pandemic) and gradually declines as regulatory frameworks mature and institutional algorithmic participation deepens.27

The integration of artificial intelligence and automated trading in these frontier markets has radically accelerated the evolutionary cycle. As more institutional participants deploy algorithmic models, the lifespan of traditional market anomalies (such as basic calendar effects or simple moving average crossovers) rapidly contracts. Consequently, the edge in frontier markets increasingly belongs to hybrid models capable of synthesizing unstructured, alternative data—such as high-frequency local news sentiment and granular macroeconomic indicators—to front-run the traditional econometric signals utilized by legacy asset managers.21

## **5\. Macroeconomic Determinants of Philippine Banking Equities**

To successfully operationalize hybrid modeling within the Philippine context, one must deeply understand the structural composition of the PSEi. The index is heavily weighted toward massive domestic holding companies and the financial sector, with universal banks acting as the primary conduits of systemic capital and liquidity. BDO Unibank (BDO) and the Bank of the Philippine Islands (BPI) consistently dominate the Philippine banking sector by market capitalization, rendering their equity returns highly sensitive to domestic monetary policy adjustments and international currency valuations.39

### **5.1 Systemic Importance and Capital Shortfall Vulnerability**

The banking sector functions as the central nervous system of the Philippine economy. According to Bangko Sentral ng Pilipinas (BSP) Discussion Paper 2025-08 authored by Godoy (2025), financial institutions maintain the absolute largest share of systemic risk within the sovereign ecosystem.41 Utilizing the SRISK metric—a sophisticated financial measure that calculates the expected capital shortfall of a specific entity conditional on a severe, prolonged market decline—the study demonstrates that the banking sector accounted for an overwhelming 58.0 to 72.0 percent of total systemic risk during the extreme volatility of the COVID-19 pandemic.41

Crucially, variance decomposition analysis reveals that the variability of this systemic risk is driven primarily by exogenous macroeconomic shocks to Gross Domestic Product (GDP), the BSP policy rate, and the USD/PHP exchange rate.41 Therefore, any predictive machine learning model forecasting the equity returns of BDO and BPI must ingest these specific macroeconomic variables not as static, linear controls, but as highly dynamic features exhibiting complex, non-linear interactions with bank-specific balance sheet health.

### **5.2 The Non-Linear USD/PHP Transmission Mechanism**

The sensitivity of Philippine banking equities to the USD/PHP exchange rate is profound, structural, and inherently multi-faceted. When the Philippine Peso depreciates against the US Dollar, the immediate impact on a universal bank's equity valuation depends heavily on its net open foreign exchange (FX) position. As delineated in BSP Discussion Paper 2022-02, a depreciation of the domestic currency generally reduces overall bank lending and acts as a severe contractionary force if the banking sector holds a negative net open FX position.43

However, top-tier universal banks like BDO and BPI operate with highly sophisticated derivative hedging mechanisms and often deliberately maintain positive net open FX positions. In these specific instances, a depreciating Peso can theoretically inflate the local currency value of their foreign-denominated assets, thereby artificially supporting capital adequacy ratios in the short term.43 Yet, this positive balance sheet effect is frequently offset by secondary, cascading macroeconomic headwinds. A sustained, uncontrolled rise in the USD/PHP rate typically accelerates imported inflation (given the Philippines' reliance on imported energy and commodities), heavily pressuring the BSP to tighten monetary policy by aggressively raising the Target Reverse Repurchase (RRP) Rate.44

### **5.3 Net Interest Margins and Remittance-Driven Liquidity**

The direct relationship between BSP policy rates and banking equity returns is historically viewed through the critical metric of Net Interest Margins (NIM). While aggressive rate hikes generally expand NIMs by allowing banks to rapidly reprice corporate and consumer loans higher, rapid monetary tightening simultaneously depresses aggregate economic growth and heightens the probability of systemic defaults, leading to an expansion in Non-Performing Loans (NPLs).46 Philippine banking executives have indicated that a 25 basis point policy rate cut could result in a 4 to 9 basis point compression in NIMs, holding all else equal.42

Crucially, the Philippine banking sector possesses a unique, structural liquidity buffer that fundamentally alters this dynamic: Overseas Filipino Worker (OFW) remittances. Unlike volatile foreign portfolio investments ("hot money"), remittances act as a massive, counter-cyclical liquidity provider.48 When the USD/PHP rate depreciates, the domestic purchasing power of dollar-denominated remittances mathematically increases. This injects robust consumer liquidity into the domestic economy, driving Current Account and Savings Account (CASA) deposit growth at major retail banks like BDO and BPI.50 This persistent, remittance-driven liquidity heavily insulates Philippine banks from external funding shocks and supports sustained loan growth even during periods of tight domestic monetary policy.

| Macroeconomic Driver | Primary Mechanism on Bank Equities | Theoretical Impact on BPI / BDO |
| :---- | :---- | :---- |
| **BSP Policy Rate Hikes** | Expansion of Net Interest Margin (NIM) | Positive in the short term; Negative long-term if NPLs surge |
| **USD/PHP Depreciation** | Valuation of positive net open FX positions | Positive balance sheet effect; Negative due to imported inflation |
| **OFW Remittance Surges** | Expansion of low-cost CASA deposit base | Highly Positive; provides cheap funding and limits liquidity risk |

Consequently, a hybrid machine learning model predicting BPI or BDO returns must meticulously map the complex, non-linear interplay between USD/PHP volatility, anticipated BSP rate adjustments, and real-time remittance flows. A classical linear OLS model would mathematically average out these effects, missing the reality that a mild depreciation coupled with strong remittance inflows is highly bullish for banking equities, whereas a severe depreciation triggering aggressive, panicked rate hikes constitutes a deeply bearish regime.

## **6\. Multi-Horizon Predictability: Signal Decay and Temporal Aggregation**

A critical, often overlooked dimension of algorithmic asset pricing is the exact forecasting horizon. The performance, architecture, and validity of any predictive model are intimately tied to the temporal alignment of its inputs and target variables. Predicting the 1-day forward return of an equity requires entirely different architectural configurations and feature engineering pipelines than forecasting a 5-day, 20-day, or 60-day holding period. This dynamic introduces the complex concept of multi-horizon predictability, requiring an exploration of how the signal-to-noise ratio fundamentally evolves as financial data is temporally aggregated.

### **6.1 Short-Term Microstructure Noise vs. Medium-Term Fundamental Trends**

Financial time series at daily or sub-daily frequencies are absolutely dominated by microstructure noise—transient, semi-random price fluctuations caused by bid-ask bounce, institutional order flow imbalances, and algorithmic execution slippage. In a 1-day forecast horizon, the variance of this statistical noise vastly exceeds the variance of the underlying fundamental signal, resulting in a microscopic Signal-to-Noise Ratio (SNR).4

When machine learning models are deployed to forecast 1-day returns, they operate primarily as high-frequency statistical arbitrage engines. They rely heavily on the mathematics of mean reversion, exploiting the tendency of assets to rapidly snap back from localized liquidity vacuums or transient order book imbalances.14 However, these short-term signals decay almost instantaneously. If an LSTM network successfully identifies a 1-day micro-trend, execution latency, bid-ask spreads, and exchange transaction costs often obliterate the theoretical algorithmic alpha before it can be economically realized in a live trading environment.7

As the forecast horizon mathematically extends to 5-day or 20-day periods, the effects of temporal aggregation fundamentally alter the underlying data structure. Short-term microstructure noise essentially cancels itself out over longer intervals, allowing persistent macroeconomic trends, factor momentum, and deep fundamental valuation shifts to dominate the return profile.53

### **6.2 Horizon Decay in Machine Learning Architectures**

The econometric and computer science literature presents deeply conflicting evidence regarding the optimal forecast horizon for machine learning models. Fischer and Krauss (2018) achieved immense statistical success forecasting 1-day directional movements using LSTMs, leaning heavily on the network's unique internal state memory to model short-term reversal profiles.3 Conversely, broader applications of deep learning to multi-horizon equity returns indicate that predictive accuracy and ![][image7] metrics generally decay significantly as the forecasting horizon lengthens.54

The mathematical phenomenon of horizon decay is rooted in the exponential accumulation of uncertainty. While extending the forecast from 1 day to 5 days successfully smooths out microstructure noise, extending it further to 30 or 90 days introduces profound, unquantifiable fundamental uncertainty. Over a long horizon, the probability of exogenous shocks—such as unannounced central bank interventions, severe geopolitical crises, or sudden corporate governance scandals—increases exponentially, entirely severing the deterministic link between the ![][image10] feature set and the ![][image11] target return.56

Recent empirical studies applying ensemble models (such as XGBoost and Ridge Regression) to emerging market equities over 5-day and 21-day horizons confirm this exact structural friction. Models tend to yield directional accuracies hovering only slightly above a random coin toss (49%–54%) and near-zero ![][image7] values, mathematically indicating that weak predictive signals are rapidly overshadowed by compounding market noise and exogenous variance as the horizon extends.52

To construct a robust, economically viable hybrid model for the Philippine Stock Exchange, quantitative practitioners must adopt a multi-head forecasting architecture. Such a framework utilizes Temporal Convolutional Networks (TCN) or self-attention mechanisms to simultaneously output predictions across multiple distinct horizons.57 This advanced architecture allows the portfolio optimization layer to execute high-turnover trades based on 1-day LSTM reversal signals, while simultaneously maintaining core, highly capitalized positional biases dictated by the 5-day and 20-day macro-driven XGBoost outputs. This dual-layered approach maximizes risk-adjusted returns net of all real-world transaction costs.

## **7\. Strategic Synthesis and Future Outlook**

The application of hybrid machine learning and econometric modeling to equity return prediction represents a critical, irreversible evolution in empirical asset pricing. As decisively demonstrated by the foundational literature authored by Gu, Kelly, and Xiu (2020), advanced algorithmic architectures such as Random Forests, XGBoost, and Long Short-Term Memory networks systematically outperform classical OLS regressions. They achieve this by computationally capturing the highly complex, non-linear, and sequential dependencies inherent in massive financial datasets. These models successfully abandon rigid linear constraints in favor of fluid, data-driven optimization, providing substantial economic gains in rigorous out-of-sample environments.

The empirical success of these algorithms in frontier and emerging environments like the Philippine Stock Exchange powerfully reinforces the Adaptive Market Hypothesis. Market efficiency is clearly not an absolute, static condition, but an ongoing evolutionary process where localized pockets of predictability—driven by institutional momentum, retail mean reversion, and cognitive behavioral biases—continuously emerge and dissipate. The strict application of the "No Free Lunch" theorem mandates that modeling architectures must be highly specialized and dynamically ensembled to survive these violently shifting volatility regimes.

For systemically crucial assets such as Philippine banking equities (specifically BDO and BPI), accurate mathematical modeling requires the seamless integration of complex macroeconomic transmission mechanisms. Stock returns in this concentrated sector are deterministic outputs of a highly interconnected, non-linear system where USD/PHP exchange rate volatility, remittance-driven liquidity buffers, and Bangko Sentral ng Pilipinas monetary policy adjustments collide to shape bank balance sheets and dictate systemic risk profiles. A model that ignores the specific dynamics of OFW remittances or net open FX positions will systematically fail to predict forward banking returns.

Ultimately, the realization of true, persistent alpha relies heavily on the precise management of multi-horizon predictability. While short-term horizons are heavily dominated by highly exploitable but extremely transient microstructure noise, medium-term horizons are governed by structural macroeconomic trends that are deeply vulnerable to expanding, compounding uncertainty. The deployment of a sophisticated, multi-horizon hybrid model—one that meticulously synthesizes deep sequence learning for high-frequency pattern recognition with powerful gradient-boosted ensembles for macro-fundamental interactions—provides the most rigorous, scientifically sound framework for navigating and extracting value from the extreme complexities of the modern Philippine financial ecosystem.

#### **Works cited**

1. NBER WORKING PAPER SERIES EMPIRICAL ASSET PRICING VIA MACHINE LEARNING Shihao Gu Bryan Kelly Dacheng Xiu Working Paper 25398 htt, accessed April 2, 2026, [https://www.nber.org/system/files/working\_papers/w25398/w25398.pdf](https://www.nber.org/system/files/working_papers/w25398/w25398.pdf)  
2. Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S\&P 500 \- EconPapers, accessed April 2, 2026, [https://econpapers.repec.org/RePEc:eee:ejores:v:259:y:2017:i:2:p:689-702](https://econpapers.repec.org/RePEc:eee:ejores:v:259:y:2017:i:2:p:689-702)  
3. Deep learning with long short-term memory networks for financial market predictions, accessed April 2, 2026, [https://ideas.repec.org/a/eee/ejores/v270y2018i2p654-669.html](https://ideas.repec.org/a/eee/ejores/v270y2018i2p654-669.html)  
4. Does Noise Hurt Economic Forecasts?, accessed April 2, 2026, [https://economics.ucr.edu/wp-content/uploads/2025/03/ssrn-4659309.pdf](https://economics.ucr.edu/wp-content/uploads/2025/03/ssrn-4659309.pdf)  
5. Machine Learning and Causality: The Impact of Financial Crises on Growth, WP/19/228, November 2019 \- International Monetary Fund, accessed April 2, 2026, [https://www.imf.org/-/media/files/publications/wp/2019/wpiea2019228-print-pdf.pdf](https://www.imf.org/-/media/files/publications/wp/2019/wpiea2019228-print-pdf.pdf)  
6. Machine Learning and Causality: The Impact of Financial Crises on Growth in \- IMF eLibrary, accessed April 2, 2026, [https://www.elibrary.imf.org/view/journals/001/2019/228/article-A001-en.xml](https://www.elibrary.imf.org/view/journals/001/2019/228/article-A001-en.xml)  
7. Empirical Asset Pricing via Machine Learning \- EconPapers, accessed April 2, 2026, [https://econpapers.repec.org/RePEc:oup:rfinst:v:33:y:2020:i:5:p:2223-2273.](https://econpapers.repec.org/RePEc:oup:rfinst:v:33:y:2020:i:5:p:2223-2273.)  
8. Autoencoder-Based Three-Factor Model for the Yield Curve of Japanese Government Bonds and a Trading Strategy \- MDPI, accessed April 2, 2026, [https://www.mdpi.com/1911-8074/13/4/82](https://www.mdpi.com/1911-8074/13/4/82)  
9. Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S\&P 500 \- EconStor, accessed April 2, 2026, [https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf](https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf)  
10. Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S\&P 500 \- IDEAS/RePEc, accessed April 2, 2026, [https://ideas.repec.org/a/eee/ejores/v259y2017i2p689-702.html](https://ideas.repec.org/a/eee/ejores/v259y2017i2p689-702.html)  
11. Empirical Asset Pricing via Machine Learning | The Review of Financial Studies | Oxford Academic, accessed April 2, 2026, [https://academic.oup.com/rfs/article/33/5/2223/5758276](https://academic.oup.com/rfs/article/33/5/2223/5758276)  
12. Empirical Asset Pricing via Machine Learning \- IDEAS/RePEc, accessed April 2, 2026, [https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2223-2273..html](https://ideas.repec.org/a/oup/rfinst/v33y2020i5p2223-2273..html)  
13. Multi-Horizon Equity Returns Predictability via Machine Learning \- Czech Journal of Economics and Finance, accessed April 2, 2026, [https://journal.fsv.cuni.cz/storage/1531\_attachment.pdf](https://journal.fsv.cuni.cz/storage/1531_attachment.pdf)  
14. Deep learning with long short-term memory networks for financial market predictions, accessed April 2, 2026, [https://econpapers.repec.org/RePEc:eee:ejores:v:270:y:2018:i:2:p:654-669](https://econpapers.repec.org/RePEc:eee:ejores:v:270:y:2018:i:2:p:654-669)  
15. Deep Learning with a Long Short-Term Memory Networks Approach for Rainfall-Runoff Simulation \- MDPI, accessed April 2, 2026, [https://www.mdpi.com/2073-4441/10/11/1543](https://www.mdpi.com/2073-4441/10/11/1543)  
16. Deep learning with long short-term memory networks for financial market predictions, accessed April 2, 2026, [https://www.semanticscholar.org/paper/Deep-learning-with-long-short-term-memory-networks-Fischer-Krauss/701e59358fda0d865d7b26cd954a93a5ad20fd13](https://www.semanticscholar.org/paper/Deep-learning-with-long-short-term-memory-networks-Fischer-Krauss/701e59358fda0d865d7b26cd954a93a5ad20fd13)  
17. Enhancing stock index prediction: A hybrid LSTM-PSO model for improved forecasting accuracy \- PMC, accessed April 2, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11731719/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11731719/)  
18. A Review of ARIMA vs. Machine Learning Approaches for Time Series Forecasting in Data Driven Networks \- MDPI, accessed April 2, 2026, [https://www.mdpi.com/1999-5903/15/8/255](https://www.mdpi.com/1999-5903/15/8/255)  
19. Simple Explanation of the No-Free-Lunch Theorem and Its Implications \- IDEAS/RePEc, accessed April 2, 2026, [https://ideas.repec.org/a/spr/joptap/v115y2002i3d10.1023\_a1021251113462.html](https://ideas.repec.org/a/spr/joptap/v115y2002i3d10.1023_a1021251113462.html)  
20. Confronting Machine Learning With Financial Research \- arXiv, accessed April 2, 2026, [https://arxiv.org/pdf/2103.00366](https://arxiv.org/pdf/2103.00366)  
21. Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S\&P 500 | Request PDF \- ResearchGate, accessed April 2, 2026, [https://www.researchgate.net/publication/309380988\_Deep\_neural\_networks\_gradient-boosted\_trees\_random\_forests\_Statistical\_arbitrage\_on\_the\_SP\_500](https://www.researchgate.net/publication/309380988_Deep_neural_networks_gradient-boosted_trees_random_forests_Statistical_arbitrage_on_the_SP_500)  
22. Behavior of calendar anomalies, market conditions and adaptive market hypothesis: Evidence from Pakistan stock exchange \- EconStor, accessed April 2, 2026, [https://www.econstor.eu/bitstream/10419/188301/1/pjcss378.pdf](https://www.econstor.eu/bitstream/10419/188301/1/pjcss378.pdf)  
23. Market Efficiency of Asian Stocks: Evidence based on Narayan-Liu-Westerlund GARCH-based Unit root test \- Munich Personal RePEc Archive, accessed April 2, 2026, [https://mpra.ub.uni-muenchen.de/109828/1/MPRA\_paper\_109828.pdf](https://mpra.ub.uni-muenchen.de/109828/1/MPRA_paper_109828.pdf)  
24. Jae Hoon Kim | IDEAS/RePEc, accessed April 2, 2026, [https://ideas.repec.org/e/c/pki102.html](https://ideas.repec.org/e/c/pki102.html)  
25. Revisiting efficiency in MENA stock markets during political shocks: evidence from a multi-step approach \- PMC, accessed April 2, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC8482441/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8482441/)  
26. Determinants of Adaptive Behaviour in Stock Market: A Review \- ResearchGate, accessed April 2, 2026, [https://www.researchgate.net/publication/370073413\_Determinants\_of\_Adaptive\_Behaviour\_in\_Stock\_Market\_A\_Review](https://www.researchgate.net/publication/370073413_Determinants_of_Adaptive_Behaviour_in_Stock_Market_A_Review)  
27. A Composite Efficiency Index for ASEAN Foreign Exchange Markets Tran Trong Huynh1,2,\*, Thi Thu Hong Dinh2, accessed April 2, 2026, [https://etamaths.com/index.php/ijaa/article/view/4592/1515](https://etamaths.com/index.php/ijaa/article/view/4592/1515)  
28. Analyse de l'efficience des marchés de taux de change de l'Afrique du nord: Approche Multifractale Analysis of exchange rate markets efficiency in North Africa: Multifractal approach \- African Scientific Journal, accessed April 2, 2026, [https://africanscientificjournal.com/index.php/AfricanScientificJournal/article/download/1207/1106/1241](https://africanscientificjournal.com/index.php/AfricanScientificJournal/article/download/1207/1106/1241)  
29. Active Portfolio Management Adapted For the Emerging Markets \- DSpace@MIT, accessed April 2, 2026, [https://dspace.mit.edu/bitstream/handle/1721.1/65814/750609635-MIT.pdf?sequence=2](https://dspace.mit.edu/bitstream/handle/1721.1/65814/750609635-MIT.pdf?sequence=2)  
30. Random walks in the different sectoral submarkets of the Philippine Stock Exchange amid modernization \- Economics and Finance Research, accessed April 2, 2026, [https://ideas.repec.org/a/phs/prejrn/v50y2013i1p57-82.html](https://ideas.repec.org/a/phs/prejrn/v50y2013i1p57-82.html)  
31. Random walks in the different sectoral submarkets of the Philippine Stock Exchange amid modernization \- ResearchGate, accessed April 2, 2026, [https://www.researchgate.net/publication/251238846\_Random\_walks\_in\_the\_different\_sectoral\_submarkets\_of\_the\_Philippine\_Stock\_Exchange\_amid\_modernization](https://www.researchgate.net/publication/251238846_Random_walks_in_the_different_sectoral_submarkets_of_the_Philippine_Stock_Exchange_amid_modernization)  
32. EFFICIENT MARKET HYPOTHESIS: CASE OF THE CROATIAN CAPITAL MARKET \- ResearchGate, accessed April 2, 2026, [https://www.researchgate.net/publication/334391730\_EFFICIENT\_MARKET\_HYPOTHESIS\_CASE\_OF\_THE\_CROATIAN\_CAPITAL\_MARKET](https://www.researchgate.net/publication/334391730_EFFICIENT_MARKET_HYPOTHESIS_CASE_OF_THE_CROATIAN_CAPITAL_MARKET)  
33. Dynamic Factor Analysis of Price Movements in the Philippine Stock Exchange, accessed April 2, 2026, [https://ideas.repec.org/p/arx/papers/2510.15938.html](https://ideas.repec.org/p/arx/papers/2510.15938.html)  
34. Momentum Strategies Across Asset Classes \- CME Group, accessed April 2, 2026, [https://www.cmegroup.com/education/files/jpm-momentum-strategies-2015-04-15-1681565.pdf](https://www.cmegroup.com/education/files/jpm-momentum-strategies-2015-04-15-1681565.pdf)  
35. Estimating the Extent of Market Herding in the Philippine Equities Market \- Bangko Sentral ng Pilipinas, accessed April 2, 2026, [https://www.bsp.gov.ph/Media\_And\_Research/Publications/BS2017\_02.pdf](https://www.bsp.gov.ph/Media_And_Research/Publications/BS2017_02.pdf)  
36. Mr. Huynh Trong Huynh | Author \- SciProfiles, accessed April 2, 2026, [https://sciprofiles.com/profile/2532858?utm\_source=mdpi.com\&utm\_medium=website\&utm\_campaign=avatar\_name](https://sciprofiles.com/profile/2532858?utm_source=mdpi.com&utm_medium=website&utm_campaign=avatar_name)  
37. A Composite Efficiency Index for ASEAN Foreign Exchange Markets \- ResearchGate, accessed April 2, 2026, [https://www.researchgate.net/publication/398085175\_A\_Composite\_Efficiency\_Index\_for\_ASEAN\_Foreign\_Exchange\_Markets](https://www.researchgate.net/publication/398085175_A_Composite_Efficiency_Index_for_ASEAN_Foreign_Exchange_Markets)  
38. BSP Economic Newsletter \- Bangko Sentral ng Pilipinas, accessed April 2, 2026, [https://www.bsp.gov.ph/Sites/researchsite/Pages/Publications/BSP-Economic-Newsletter.aspx](https://www.bsp.gov.ph/Sites/researchsite/Pages/Publications/BSP-Economic-Newsletter.aspx)  
39. (PDF) Valuation Metrics, Market Efficiency, and Investor Sentiment: A Descriptive Analysis of Philippine Stock Exchange–Listed Firms \- ResearchGate, accessed April 2, 2026, [https://www.researchgate.net/publication/399328096\_Valuation\_Metrics\_Market\_Efficiency\_and\_Investor\_Sentiment\_A\_Descriptive\_Analysis\_of\_Philippine\_Stock\_Exchange-Listed\_Firms](https://www.researchgate.net/publication/399328096_Valuation_Metrics_Market_Efficiency_and_Investor_Sentiment_A_Descriptive_Analysis_of_Philippine_Stock_Exchange-Listed_Firms)  
40. Preliminary Offer Supplement 04082022 vF \- Ayala Corporation, accessed April 2, 2026, [https://ayala.com/app/uploads/2023/05/Ayala-Corporation-Final-Offer-Supplement-w-financials-05102022-1.pdf](https://ayala.com/app/uploads/2023/05/Ayala-Corporation-Final-Offer-Supplement-w-financials-05102022-1.pdf)  
41. Measuring the Systemic Risk Contribution of Philippine Industries and Conglomerate Groups \- Bangko Sentral ng Pilipinas, accessed April 2, 2026, [https://www.bsp.gov.ph/Sites/researchsite/Publications/BSP-Discussion-Papers/DP202508.pdf](https://www.bsp.gov.ph/Sites/researchsite/Publications/BSP-Discussion-Papers/DP202508.pdf)  
42. Philippines Strategy \- 2025 Year Ahead, accessed April 2, 2026, [https://mkefactsettd.maybank-ke.com/PDFS/427202.pdf](https://mkefactsettd.maybank-ke.com/PDFS/427202.pdf)  
43. BSP DISCUSSION PAPER \- Bangko Sentral ng Pilipinas, accessed April 2, 2026, [https://www.bsp.gov.ph/Sites/researchsite/Publications/BSP-Discussion-Papers/DP202202.pdf](https://www.bsp.gov.ph/Sites/researchsite/Publications/BSP-Discussion-Papers/DP202202.pdf)  
44. Second Quarter 2025 \- Bangko Sentral ng Pilipinas, accessed April 2, 2026, [https://www.bsp.gov.ph/Lists/Quarterly%20Report/Attachments/27/LTP\_2qtr2025.pdf](https://www.bsp.gov.ph/Lists/Quarterly%20Report/Attachments/27/LTP_2qtr2025.pdf)  
45. FEBRUARY 2026 \- Bangko Sentral ng Pilipinas, accessed April 2, 2026, [https://www.bsp.gov.ph/Price%20Stability/MonetaryPolicyReport/FullReport-February2026.pdf](https://www.bsp.gov.ph/Price%20Stability/MonetaryPolicyReport/FullReport-February2026.pdf)  
46. Important notice \- PDS Group, accessed April 2, 2026, [https://www.pds.com.ph/wp-content/uploads/2025/12/aa.-BPI-Fixed-Rate-Bonds-Due-2025-Pricing-Supplement-13Nov20231.pdf](https://www.pds.com.ph/wp-content/uploads/2025/12/aa.-BPI-Fixed-Rate-Bonds-Due-2025-Pricing-Supplement-13Nov20231.pdf)  
47. BPI Php 200 Bn Bond Program (2025).pdf, accessed April 2, 2026, [https://www.bpi.com.ph/content/dam/bau/personal-banking/investments/pdfs/BPI%20Php%20200%20Bn%20Bond%20Program%20(2025).pdf?download=true](https://www.bpi.com.ph/content/dam/bau/personal-banking/investments/pdfs/BPI%20Php%20200%20Bn%20Bond%20Program%20\(2025\).pdf?download=true)  
48. accessed April 2, 2026, [https://openknowledge.worldbank.org/bitstreams/b78ca1d3-5a64-56f7-b291-981294fed46d/download](https://openknowledge.worldbank.org/bitstreams/b78ca1d3-5a64-56f7-b291-981294fed46d/download)  
49. A note on the effectiveness of intervention in the foreign exchange market: the case of the Philippines \- Bank for International Settlements, accessed April 2, 2026, [https://www.bis.org/publ/bppdf/bispap73s.pdf](https://www.bis.org/publ/bppdf/bispap73s.pdf)  
50. 1\. How is the exchange rate defined? The exchange rate is the price of a unit of foreign currency \- Bangko Sentral ng Pilipinas, accessed April 2, 2026, [https://www.bsp.gov.ph/Media\_and\_Research/Primers%20Faqs/exchange.pdf](https://www.bsp.gov.ph/Media_and_Research/Primers%20Faqs/exchange.pdf)  
51. Noise in Expectations: Evidence from Analyst Forecasts | The Review of Financial Studies, accessed April 2, 2026, [https://academic.oup.com/rfs/article-abstract/37/5/1494/7461199](https://academic.oup.com/rfs/article-abstract/37/5/1494/7461199)  
52. Stock Return Prediction on the LQ45 Market Index in the Indonesia Stock Exchange Using a Machine Learning Algorithm Based on Technical Indicators \- MDPI, accessed April 2, 2026, [https://www.mdpi.com/1911-8074/18/12/714](https://www.mdpi.com/1911-8074/18/12/714)  
53. Forecasting Stock Market Volatility Using Support Vector Regression Across Daily, Weekly and Monthly Data Frequencies \- Diva-portal.org, accessed April 2, 2026, [https://www.diva-portal.org/smash/get/diva2:1886186/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:1886186/FULLTEXT01.pdf)  
54. econstor, accessed April 2, 2026, [https://www.econstor.eu/bitstream/10419/247369/1/wp2021-02.pdf](https://www.econstor.eu/bitstream/10419/247369/1/wp2021-02.pdf)  
55. MULTI-HORIZON EQUITY RETURNS PREDICTABILITY VIA MACHINE LEARNING \- Institut ekonomických studií, accessed April 2, 2026, [https://ies.fsv.cuni.cz/sites/default/files/uploads/files/wp\_2021\_02\_nechvatalova.pdf](https://ies.fsv.cuni.cz/sites/default/files/uploads/files/wp_2021_02_nechvatalova.pdf)  
56. Multi-horizon short-term load forecasting using hybrid of LSTM and modified split convolution \- PMC, accessed April 2, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10557505/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10557505/)  
57. Deep Learning-Based Hybrid Model with Multi-Head Attention for Multi-Horizon Stock Price Prediction \- MDPI, accessed April 2, 2026, [https://www.mdpi.com/1911-8074/18/10/551](https://www.mdpi.com/1911-8074/18/10/551)  
58. An Evaluation of Deep Learning Models for Stock Market Trend Prediction \- arXiv, accessed April 2, 2026, [https://arxiv.org/html/2408.12408v1](https://arxiv.org/html/2408.12408v1)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAAB+ElEQVR4Xu2WO0gdQRSGjyEm+GjEJmIRBBERLHyEKGpiZUQhoJVYiBjUziohREVIbMVGFCFiG1IIgYCFiBoSEa0ELawEK8FaSNKo/8+cvTuZ3F2Xq+JcuB98LHtmYM/snD2zIjlyZEQt7FK79doJO2C9Nc87sjbxXrigXsJN+AFOwy9wF/alZntGv8rEXzljH+EFrFS9YlH9CwudsSkxCxpSveJI/ekOgH14AotUbygT80Ypy4LkwWfwF9yCVRr3iqC26Te4DL+LqetPYhbhJUvwt/rYif+BT6yYDTtN3KIeuIGEVMNmNZasTfwYbqg27CDp2iNh55mT6OTy4YEbVKKSegRn4J6Yc4RGUiEmObY8avNZx5qceBJa4aobVN6qUTD5axMfFpNcm2rD1sixGlgCR9UWOA8HwqkpBtVtuCbp+/47NYrYxF/CH/BMTB2zV9MJaw4fys7CFsmjv1F9I+bB9lwXll2Ddc/+X6xOqsE9fRhOjU88KSyl17DciR/Cp06M8OOmpxLWP5MakXDHVtTgntbpXMLEx9Vb5TlcF/P3WAoLNEbaVZ4DTLhH4zZJSoW7GbejGfECfpXw4Uz6XMwi+OYoFzYr/+8UiUucZcm/0R31/b/DNyNrEycsD5sxMZ0nIO5njB83vXd4cES9Qa9h93B34E64AoM7cyIys/8xAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB4AAAAZCAYAAAAmNZ4aAAABcUlEQVR4Xu3WvStFcRzH8a+HhZukGETCIIuyKCUlj4tYxECUlFLKimJQshkxmEwMymDULSUPhcEfYfBQJMnC59vvc+45fp1L8fs5A+961b3ne+rXebxX5L+/VgN1QQd0kpaibmiLcNIAbcIb3MMcZxV0xtkFzHDmtBN4hmJ+L6Q0NAU7+WhczJFNQR7sUEt0Jx/p9XyES1iDfvJeYgtrG2JOd3CD/Vq6oC68bg981ivmiM/hQcLnOK4yaKfPyrU32DXDtpi7eVLMUetdruJqhVGKa54G7UFQPe2LeWa1IniCY/pOp1RrD/QU6tFcUenHceZNphqt2TDsQiVFq4JVuKFlKInukMjCI3ALL/BKs8EQLcAd5+pazEJaOfTAoYSXyU7nW+S0GjE/HNnSo5wgpy3CNAyRpqe4mp+PoI76IJ/bf9wSrIg5pUrTS3XAz3sSPk5j3OasAnuDmIW0HAl/Up2X2MJ2+kfhq9enl7K90zO9A7ijUvPgNYHNAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAv0lEQVR4Xu3QvwpBUQDH8ePPosjE4gkUeQCzMvICFhllspotHsH1BLLKZjSYDFIGm4GUJ+D3c39Hx3DqDMriW5+63fu7de8x5t9PKkJFWA4akrKjLEzhJGOYQF/WkOFwAFVYyhwSfKD20ORFXQ9u0nZG7ApDXgQPWRkewh9jJeG998tdOIqtJ2cTn8KrCDbC0rCVjh2xA6xkBgsNPkYsaFgw8QfzLClvnG9ya8EdkuJtBxcYibea8CzJW/Dwuz0Brjcs91kkcuQAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAYCAYAAADOMhxqAAAAp0lEQVR4Xu3QIQ9BURiH8TOKoBAkm2SmYNN0H0IRUYiiJPgC2u0okiLT6TYU3TfgObv/e/ZqNOE+22935773nLNd59L+qRIq4sughVr4wlTABieZIsIQe8wlNEEDK3m4+EZfH1cJdfS8ydjMFjhI6OcNvjJe0jTvz5jJRz08xf8hX9XFB9Sli5xmbomdJI1wR1bWZuaOGEhSGxdsxa9DRbsw5Y20r3sDFWQkjpXpCZAAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAH0AAAAXCAYAAAAm70AZAAAEMElEQVR4Xu2aaWgVVxTHj602VmpVqrgUEawYl1aLxfWDcatRqaKIgvrB9INWVPpBQUFRAiKCuGG/FBFFVCyC4C5SlxRaizRSFIViRaRIVdwQ9wX1/HPPZO6czJ13X/Lmocn84AfmnJc3M/87y50biTIyUqIbu9hyhfoZffiu0oGdqIsFIi6buHxS5VNxsG4UkCW6kAKd2X66mCcLxX3sKdVLC2RTjHxqacGeFL9UvWZsOTuGHS1+yw6UfkfpjRDLpB6H70G1Y8dR3W1+If3e7Cgy2xrJ9pc6aM5WsV2sWn0ZT8mDjmx0PnHZ+OSTNOglZPYlOGaIfw+wPyQ/29vCvnxs9SNsYTeJmlbsSvZ39o14kZ0ufVxVV6R+lswV4sJ1UJpe7Gr2KoXbPMZ+I/2p7FP2OXuITOg2w9mjqlYfcg06stH5xGXjk0/SoH/GLmWrKZrHBPtDZL4bvbviOrZl5BMCduwm20Z00ZV9Jf5h1XE7/ZMttWouXAflYgaFB4mTIABX1GG2rVXT7GJ/UDVcMfATh7hqbXINuk2QT1w2PvkkDXrAWArz2K564EcKBzp2sAOyQW+Cg76NXa6LDvaL2OhXZJ69v7I97A8lkOugNB+xt8QbZOYeeG4doYRnlYCQH5L5nQDMEyBOhjjbhx+tAYN+WtWScGXjk4/PoH/I/i8+oOjATiIzll4gmMm66KBcDM40hG9PonKR66DiWCNim8vY45R8RwpAQC/YPrqRBxj0Kl1MoKHZ+OSzXsR2pkkNE8cDFD3BnWB2iV/uqxsOgtlqMMHC7SYfFuiCB93F1+x9Mvvsyz8UBpMPFeJe9j8yJ10nq++iodn45PO1iG0cJJMN7ig+F0INQ8mEmXj/t/hAxHMOG50TbacCZq7wDpl97RltJ4KZfaUupkgxs8FbAu5kv5GZT3gzi8yZ7MtPIiYMz9hz0XbBweQKr14QrywIE7c2XzaSuVqLRTGz2UAmj7wX075jb+uig1XsfBHsJLPRQbWfKCwlZCZG+H6Iq+gamfdP3zvTz+xuXUyJIB+QdjbgBOV3wdaC5wF2LunVBywisy5sM4zCSUuhwSTsFzKrSzaYyGGbs1XdBVYYMRhpo/NJMxtcDPAJu0P1vMDE4zGFy4YaLHdilW6PbggXyKyM6Ved+oL9wbIqbufzVA9gEveS/Us3HOBKqNDFAhFk48rHzqZQ+QAsv0KcVBXRlj/V7ExdZM6QeTbBe1T39QOvCI/I9BHu1mg7b7D0+i+Z78Py6mUy6+gBn7N/Sx9eouTZLt7jMfFL4xZrZxOXj86mUPngln5dxHfj7WSz/SFf1pKZgDQ2ytjzuphhyAa9CYK/EuGdr1RsDGCmj+f+97qRETKEzLIhbAzMJbNahcHPSKBSnBItv3fgvzjhVa21bmS4wX+Zep/Bwk2uv8BlZDQN3gIW5A4oO/RUnAAAAABJRU5ErkJggg==>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAYCAYAAAARfGZ1AAABUUlEQVR4Xu3UzytEURwF8O9QSmkssFCy8A+MWNtg4Udh9iJlbWEhVnb2FrOwmJpSykpkaYNslIiytEdiJUrMOd3zevfdGeQxZTGnPs3M/d6+c9/t3mf2j5OBIZiHDvmzLMIATMCdtCVmeOmXMRjX5wgMQ4/4uYZJfT+U5bicTE2bT8sWfMA2rMAaHMgO9Gp+Vp9NcC9c1Jfhv7N5d1hATuAqGJuFkjQkKlWyDzfhoHIE79Cu3zlYN9eUWjVekWjCExSDWqe8mlshwydbhRbokgXVKtIn3JIZjTVC3tyTUMFcM+bM3FwfD0HV8NwSJ22aW+ExvMGopM6e+PvNbbqAc0mdmjVnk0cpJUu2YfGe8kz/OLyZUYO5oHYJz5IqSxY3D685j9+tMIPmFvNtOJEX4wFe5NTcrYvCbeGJIb4edqHZq/8qfG/zJtKUpdz3eur5PGW7pVZrXLoKRwAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,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>

[image9]: <data:image/png;base64,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>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAYCAYAAAAlBadpAAAA+0lEQVR4Xu3ST4sBcRzH8V/+hWxaThTRXvayF21tyg3PwMHFwUV5CopytHJwdNAWW9s+ABfOPANOnoRah03a/Yz5TMYXI83Fwbtexff3nRpjlLpnqxfyyYNLxWFLKXF2sTL8kEucnS0ICfiCKSXBb9o5WwWGsIEZDeDVvGSV9nv/IE1XVYI1uMlcFKqQoaNsXdyHiRwiL4whBEVqHmygJdRN398govTlHmcemhtLWgGlP6y80v8ezTc4oQHv+9VdKyXegxF90BPnbWgZS0x7NjEx292mrAYdMfuFBzE7WRY++fmRFvtj6xxKf/sK0KXcwYZFti42eoYw3UD/tt0rEFiby7cAAAAASUVORK5CYII=>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAYCAYAAADzoH0MAAAA90lEQVR4XmNgGAVUB1pQzIMuQQwQB+I/UGyLJkcUiALin1DMiSaHF/ADsQIQLwLic1CsyECCN9KAeDEDxOZTUAwyjCRvSALxfyB2hGKSQSQDwu/o/vcH4kNAbIQmjgIoNmAmA0QRLvAYiLnRBZHBTSBuQeIbMkBiBgQ0gfgYQgoTcDFAAtAbiNmheBUQs0HlM4H4KBCnAvE+KJaFysHBJiDeAcTzoBhkKwysAOISKHsiFIPCDAOAohIbeMSASFQ3oFgeIY0fqAHxcSgb5CpYQjMHYmmYInwgHojroWw9IN4MxRVwFQQAxQaAYoQRiQ9KCyCMLDbAAACaoC5IuZ2p7wAAAABJRU5ErkJggg==>