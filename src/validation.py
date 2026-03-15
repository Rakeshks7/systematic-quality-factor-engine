import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AlphaValidator")

class AlphaValidator:
    def __init__(self, portfolio_returns: pd.Series, ff5_data: pd.DataFrame):
        self.portfolio_returns = portfolio_returns
        self.ff5 = ff5_data

        self.aligned_data = pd.concat([self.portfolio_returns, self.ff5], axis=1).dropna()
        self.aligned_data.rename(columns={self.aligned_data.columns[0]: 'Port_Ret'}, inplace=True)

    def run_fama_french_regression(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        logger.info("Executing Fama-French 5-Factor Regression...")
        
        excess_returns = self.aligned_data['Port_Ret'] - self.aligned_data['RF']

        X = self.aligned_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        X = sm.add_constant(X) # The constant becomes our Alpha
        
        model = sm.OLS(excess_returns, X).fit()
        return model

    def calculate_performance_metrics(self, trading_days: int = 252) -> Dict[str, Any]:
        logger.info("Calculating annualized performance metrics...")
        
        returns = self.aligned_data['Port_Ret']
        rf_rate = self.aligned_data['RF'].mean() * trading_days
        
        annualized_return = returns.mean() * trading_days
        annualized_vol = returns.std() * np.sqrt(trading_days)

        excess_ret = annualized_return - rf_rate
        sharpe_ratio = excess_ret / annualized_vol if annualized_vol != 0 else 0

        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        return {
            "Annualized Return": f"{annualized_return:.2%}",
            "Annualized Volatility": f"{annualized_vol:.2%}",
            "Sharpe Ratio": round(sharpe_ratio, 2),
            "Max Drawdown": f"{max_drawdown:.2%}"
        }
