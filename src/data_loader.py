import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataLoader")

class QuantDataLoader:
    
    @staticmethod
    def fetch_fama_french_5f(start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Fetching Fama-French 5-Factor data from {start_date} to {end_date}.")

        ff_dict = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench', start_date, end_date)

        ff_data = ff_dict[0]

        ff_data = ff_data / 100.0
        
        return ff_data

    @staticmethod
    def fetch_daily_prices(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        logger.info(f"Fetching daily pricing for {len(tickers)} assets.")
        
        px_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

        px_data = px_data.ffill()
        
        return px_data
