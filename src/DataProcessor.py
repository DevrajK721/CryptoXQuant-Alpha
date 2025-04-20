# Data Processing Class 

# Core Imports 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 

# Additional Imports 
from binance.client import Client as BC 
import json 
from tqdm import tqdm
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller, kpss
from typing import * 

class DataProcessor:
    def __init__(self, secrets_path: str = '../secrets/secrets.json'):
        # Verify existence of secrets.json file 
        if not os.path.exists(secrets_path):
            raise FileNotFoundError(f"secrets.json file not found at {secrets_path}. Please ensure the provided path is correct (Hint: Use `pwd` in command line to verify current directory)")
        else:
            self.secrets_path = secrets_path
            print(f"secrets.json file found at {secrets_path}. Beginning initialization of DataProcessor class.")
        
        # Load secrets from the JSON file
        with open(self.secrets_path, 'r') as file:
            vals = json.load(file)
            self.binance_api_key = vals['BINANCE_API_KEY']
            self.frequency = vals['Trading Frequency (Yearly/Monthly/Weekly/Daily/Hourly/Minutely)']
            self.binance_api_secret = vals['BINANCE_API_SECRET']
            self.end_date = vals['Ending Date (YYYY-MM-DD)']
            self.start_date = vals['Starting Date (YYYY-MM-DD)']
            self.base_currency = vals['Base Currency']
            self.n = 50 # Fetch at least 50 data points, for worst case. 
            self.tickers = vals['Tickers of Interest']

        # Check whether all information has been loaded correctly 
        if not all([self.binance_api_key, self.binance_api_secret, self.start_date, self.end_date, self.base_currency, self.tickers]):
            raise ValueError("One or more required fields are missing in the secrets.json file.")
        
        # Initialize the Binance Client
        self.binance_client = BC(self.binance_api_key, self.binance_api_secret)
        print("Binance client initialized successfully.")

        print("All required fields loaded successfully from secrets.json.")
        print(f"{len(self.tickers)} tickers loaded successfully.")
        print(f"Frequency: {self.frequency}")
        print(f"Starting date: {self.start_date}")
        print(f"Ending date: {self.end_date}")
        print(f"Base currency: {self.base_currency}")
        print(f"Tickers: {self.tickers}")
        print("Initialization of DataProcessor class completed successfully.")

        if self.frequency == 'Daily':
            interval = BC.KLINE_INTERVAL_1DAY
        elif self.frequency == "Minutely":
            interval = BC.KLINE_INTERVAL_1MINUTE
        elif self.frequency == 'Hourly':
            interval = BC.KLINE_INTERVAL_1HOUR
        elif self.frequency == 'Weekly':
            interval = BC.KLINE_INTERVAL_1WEEK
        elif self.frequency == 'Monthly':
            interval = BC.KLINE_INTERVAL_1MONTH
        elif self.frequency == 'Yearly':
            interval = BC.KLINE_INTERVAL_1YEAR
        else:
            raise ValueError("Invalid frequency. Choose from 'Daily', 'Weekly', 'Monthly', or 'Yearly' and update in secrets.json file.")

        # Convert tickers to trading pairs 
        self.tickers = [f"{ticker}{self.base_currency}" for ticker in self.tickers]
        
        # Fetch all the crypto data requested 
        self.crypto_data = {} # Historical Data 
        self.crypto_live_data = {} # Live Data (Last n points)
        self.transforms = {} # Transformation done to data for stationarity
        # Find the path to the project root 
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        for ticker in tqdm(self.tickers, desc="Fetching Crypto Data", unit="pair",
            ncols=80, bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="blue", leave=True, dynamic_ncols=True):
            # Check if the ticker data already exists
            ticker_file_path = os.path.join(data_dir, f"{ticker}.csv")
            if os.path.exists(ticker_file_path):
                print(f"Data for {ticker} already exists at {ticker_file_path}. Skipping download.")
                # Load existing data
                data = pd.read_csv(ticker_file_path)
                self.crypto_data[ticker] = data
                continue
            
            symbol = ticker 
            columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 
            'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 
            'Taker Buy Quote Asset Volume', 'Ignore']
            data = pd.DataFrame(self.binance_client.get_historical_klines(symbol, interval, self.start_date, self.end_date), columns=columns)
            data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
            data['Close'] = data['Close'].astype(float)
            data['Open'] = data['Open'].astype(float)
            data['High'] = data['High'].astype(float)
            data['Low'] = data['Low'].astype(float)
            data['Volume'] = data['Volume'].astype(float)
            
            # Keep Open Time and Close for log returns calculation
            data = data[['Open Time', 'Close']]
            data.dropna(inplace=True)

            raw = data['Close'].copy()

            # run through your make_stationary to get the ops and final series
            stat_series, ops = self.make_stationary(raw)
            self.transforms[ticker] = ops

            # build all intermediate columns
            intermediate = {}
            series = raw.copy()
            for i, op in enumerate(ops):
                if op == 'log':
                    series = np.log(series)
                elif op == 'diff':
                    series = series.diff()
                name = "-".join(ops[: i+1])       # e.g. "log", then "log-diff", then "log-diff-diff"
                intermediate[name] = series

            # attach them to the DataFrame
            for name, ser in intermediate.items():
                data[name] = ser

            # ensure we always have a covariance-ready log-diff
            if 'log-diff' in intermediate:
                data['Covariance-LogDiff'] = intermediate['log-diff']
            else:
                # if your series was already stationary with only log, then diff once:
                data['Covariance-LogDiff'] = np.log(raw).diff()

            # finally, store the last intermediate as "Stationary-Close"
            last_name = "-".join(ops)
            data['Stationary-Close'] = intermediate[last_name]

            data.dropna(subset=['Stationary-Close'], inplace=True)
            data.drop(columns=['Close'],                         inplace=True)   # remove raw Close
            data.rename(columns={'Stationary-Close': 'Close'},    inplace=True)   # now one Close

            data = self.compute_technical_indicators_and_lags(data)
            data.dropna(inplace=True)

            data.to_csv(ticker_file_path, index=False)
            self.crypto_data[ticker] = data
            print(f"Data for {ticker} saved successfully to {ticker_file_path} with transforms {self.transforms[ticker]}.")
        
        print("All crypto data and indicators fetched/computed successfully and transformed to be stationary.")
        # Make sure crypto_data isn't empty
        if not self.crypto_data:
            print("Warning: No crypto data was successfully processed.")
        
        print(f"Fetching the last {self.n+1} data points for live data usage.")

        for ticker in tqdm(self.tickers,
                        desc="Fetching Live Data", unit="pair",
                        ncols=80, bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                        colour="red", leave=True, dynamic_ncols=True):

            ops = self.transforms.get(ticker, [])
            # count how many diffs you need to “recover” self.n points afterward
            num_diffs = ops.count('diff')
            raw_limit  = self.n + num_diffs

            try:
                klines = self.binance_client.get_klines(
                    symbol=ticker,
                    interval=interval,
                    limit=raw_limit
                )
                closes = np.array([float(k[4]) for k in klines])
                # turn into a pandas series so we can diff() easily
                s = pd.Series(closes)

                # re‑apply your recorded transforms in order
                for op in ops:
                    if op == 'log':
                        s = np.log(s)
                    elif op == 'diff':
                        s = s.diff()
                    else:
                        raise ValueError(f"Unknown transform '{op}' for {ticker}")

                # drop the NaNs created by differencing
                s = s.dropna().reset_index(drop=True)

                if len(s) != self.n:
                    print(f"Warning: After transforms, {ticker} has {len(s)} points (expected {self.n}).")
                
                # store the processed window
                self.crypto_live_data[ticker] = s.values

            except Exception as e:
                print(f"Error fetching/applying transforms for {ticker}: {e}")
                continue
        
        print("All live data fetched successfully.")
        print("Data processing completed successfully.")
        
    def is_stationary(self, s, signif: float = 0.05):
        # ADF Test
        adf_p = adfuller(s.dropna())[1]

        # KPSS Test 
        kpss_p = kpss(s.dropna(), regression='c', nlags='auto')[1]

        return (adf_p < signif) and (kpss_p > signif)
    
    def make_stationary(self, s, max_diff: int = 5):
        ops = []
        # log once
        s = np.log(s)
        ops.append('log')

        # diff until stationary
        diffs = 0
        while (not self.is_stationary(s)) and diffs < max_diff:
            s = s.diff()
            ops.append('diff')
            diffs += 1

        # drop the NaNs and return alongside the operations for reconstruction
        return s.dropna(), ops

    def compute_technical_indicators_and_lags(self, df: pd.DataFrame) -> pd.DataFrame:        
        # Compute EMA Exponential Moving Averages (EMA's) 
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # Relative Strength Index (RSI) (14)
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands (BB) (20, ±2σ)
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = rolling_mean + 2 * rolling_std
        df['BB_lower'] = rolling_mean - 2 * rolling_std
        
        # Rate of Change (ROC) (10)
        df['ROC_10'] = df['Close'].pct_change(periods=10)
        
        # Rolling Volatility (RV) (20-period std of returns)
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
        
        # Moving-Average-Convergence-Divergence (MACD) (12,26) and signal (9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Lags (k) 1 through 30
        for lag in range(1, 31):
            df[f'Lag_{lag}'] = df['Close'].shift(lag)
        
        return df

if __name__ == "__main__":
    dp = DataProcessor()  # One-Liner is all it takes to initialize the DataProcessor class)
    # For example, let's print "BTCUSDT" data
    print(dp.crypto_data['BTCUSDT'].head())  # Print the first 5 rows of BTCUSDT data
    print(dp.crypto_live_data['BTCUSDT'])  # Print the live data for predictions