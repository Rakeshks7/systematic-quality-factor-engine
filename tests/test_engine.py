import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.factor_engine import FactorInvestingPipeline

@pytest.fixture
def sample_data():
    dates = pd.date_range('2023-01-01', '2023-06-01', freq='D')
    tickers = ['AAPL', 'MSFT']

    iterables = [['2022-12-31', '2023-03-31'], tickers]
    index = pd.MultiIndex.from_product(iterables, names=['date', 'ticker'])
    fundamentals = pd.DataFrame({
        'net_income': [100, 150, 110, 160],
        'total_assets': [1000, 1500, 1100, 1600],
        'total_debt': [500, 400, 550, 450],
        'total_equity': [500, 1100, 550, 1150]
    }, index=pd.to_datetime(index.get_level_values(0)).to_frame().set_index(0).set_index(index))

    prices = pd.DataFrame(np.random.rand(len(dates), 2) + 100, index=dates, columns=tickers)

    ff5 = pd.DataFrame(np.random.rand(len(dates), 6) / 100, index=dates, 
                       columns=['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
    
    return fundamentals, prices, ff5

def test_prevent_lookahead_bias(sample_data):
    fundamentals, prices, ff5 = sample_data
    engine = FactorInvestingPipeline(fundamentals, prices, ff5)

    lagged_df = engine._prevent_lookahead_bias(lag_days=90)

    may_15_data = lagged_df.loc['2023-05-15'].loc['AAPL']

    assert may_15_data['net_income'] == 100.0
    assert may_15_data['total_assets'] == 1000.0