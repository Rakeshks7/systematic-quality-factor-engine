# Systematic Quality Factor Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade systematic equity pipeline designed to harvest the Quality risk premium. This project moves beyond simple stock-picking to construct a mathematically rigorous, dollar-neutral, long-short portfolio, validated against established asset pricing models.

##  Abstract

Institutional asset management relies on systematic factor exposure rather than discretionary selection. This engine systematically identifies high-quality equities (high profitability, low leverage) and constructs a portfolio that isolates this specific factor premium. 

Key architectural features include strict mitigation of look-ahead bias via programmatic data lagging, cross-sectional standardization to neutralize macroeconomic regime shifts, and Alpha validation using OLS regression.

##  Mathematical Framework

### 1. Factor Construction ($Q$)
The Quality factor is constructed using a composite Z-score of Return on Assets (ROA) and the Debt-to-Equity (D/E) ratio. By standardizing the cross-section of equities daily, the model isolates relative firm quality independent of market conditions:

$$Z_{Quality} = Z_{ROA} - Z_{D/E}$$

### 2. Alpha Validation
To ensure the strategy generates true excess returns ($\alpha$) and is not passively capturing broad market beta or known anomalies, the portfolio's daily excess returns are regressed against the Fama-French 5-Factor model:

$$R_p - R_f = \alpha + \beta_{MKT}(R_m - R_f) + \beta_{SMB}SMB + \beta_{HML}HML + \beta_{RMW}RMW + \beta_{CMA}CMA + \epsilon$$

A statistically significant, positive $\alpha$ confirms the successful harvesting of the Quality premium.



##  Tech Stack & Features

* **Core Engine:** `pandas`, `numpy` (Vectorized cross-sectional operations)
* **Statistical Validation:** `statsmodels` (OLS Regression for Fama-French)
* **Performance Analytics:** `alphalens` (Tear sheets, quantile analysis, forward return calculations)
* **Data Hygiene:** Automated 90-day lagging of fundamental accounting data to strictly prevent **look-ahead bias**.

##  Disclaimer

For educational and portfolio demonstration purposes only. The code and financial models provided in this repository do not constitute financial advice, investment recommendations, or an offer to buy/sell securities. Systematic trading involves substantial risk. Past performance of any factor model is not indicative of future results.