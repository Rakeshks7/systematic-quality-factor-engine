import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from typing import Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SmartBetaEngine")

class FactorInvestingPipeline:

    def __init__(self, fundamental_data: pd.DataFrame, price_data: pd.DataFrame, ff5_data: pd.DataFrame):
        self.fundamentals = fundamental_data
        self.prices = price_data
        self.ff5 = ff5_data
        self.portfolio_returns = None

    def _prevent_lookahead_bias(self, lag_days: int = 90) -> pd.DataFrame:
        logger.info(f"Applying {lag_days}-day lag to fundamental data to prevent look-ahead bias.")
        df = self.fundamentals.reset_index()
        df['date'] = df['date'] + pd.Timedelta(days=lag_days)
        df.set_index(['date', 'ticker'], inplace=True)
        daily_fundamentals = df.unstack(level=1).resample('D').ffill().stack()
        return daily_fundamentals

    def construct_quality_factor(self) -> pd.DataFrame:
        logger.info("Constructing composite Quality Factor.")
        safe_fundamentals = self._prevent_lookahead_bias()

        safe_fundamentals['ROA'] = safe_fundamentals['net_income'] / safe_fundamentals['total_assets']
        safe_fundamentals['DE_Ratio'] = safe_fundamentals['total_debt'] / safe_fundamentals['total_equity']

        def z_score(x):
            return (x - x.mean()) / x.std()

        safe_fundamentals['Z_ROA'] = safe_fundamentals.groupby(level=0)['ROA'].apply(z_score)
        safe_fundamentals['Z_DE'] = safe_fundamentals.groupby(level=0)['DE_Ratio'].apply(z_score)
        safe_fundamentals['Quality_Score'] = safe_fundamentals['Z_ROA'] - safe_fundamentals['Z_DE']
        
        return safe_fundamentals[['Quality_Score']].dropna()

    def build_dollar_neutral_portfolio(self, factor_scores: pd.DataFrame, quantiles: int = 5) -> pd.Series:
        logger.info(f"Building dollar-neutral portfolio using {quantiles} quantiles.")
        forward_returns = self.prices.pct_change().shift(-1).unstack().reorder_levels([1, 0])
        forward_returns.index.names = ['date', 'ticker']
        forward_returns.name = 'forward_return'

        data = factor_scores.join(forward_returns).dropna()
        data['Quantile'] = data.groupby(level=0)['Quality_Score'].transform(
            lambda x: pd.qcut(x, quantiles, labels=False, duplicates='drop')
        )

        top_q = quantiles - 1
        bottom_q = 0

        def calculate_leg_returns(df, quantile_label, weight_sign):
            leg = df[df['Quantile'] == quantile_label]
            if leg.empty:
                return 0.0
            weights = weight_sign * (1.0 / len(leg))
            return (leg['forward_return'] * weights).sum()

        daily_returns = data.groupby(level=0).apply(
            lambda x: calculate_leg_returns(x, top_q, 1.0) + calculate_leg_returns(x, bottom_q, -1.0)
        )
        
        self.portfolio_returns = daily_returns
        return daily_returns

    def evaluate_fama_french(self) -> sm.regression.linear_model.RegressionResultsWrapper:
        logger.info("Running Fama-French 5-Factor OLS Regression.")
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated. Run build_dollar_neutral_portfolio first.")

        aligned_data = pd.concat([self.portfolio_returns, self.ff5], axis=1).dropna()
        port_ret = aligned_data[0] 
        excess_returns = port_ret - aligned_data['RF']
        
        X = aligned_data[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
        X = sm.add_constant(X) 
        
        model = sm.OLS(excess_returns, X).fit()
        logger.info(f"Regression Alpha: {model.params['const']:.6f} (p-value: {model.pvalues['const']:.4f})")
        
        return model