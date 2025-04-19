# Data Processing Class 

# Core Imports 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import os 

# Additional Imports 
import QuantCoreStats as qcs
from binance.client import Client as BC 
import json 
from tqdm import tqdm
from datetime import datetime, timedelta
from typing import * 
from statsmodels.tsa.stattools import acf as statsmodels_acf

class DataProcessor:
    def __init__(self, secrets_path: str = '../secrets/secrets.json'):
        # Verify existence of secrets.json file 
        if not os.path.exists(secrets_path):
            raise FileNotFoundError(f"secrets.json file not found at {secrets_path}. Please ensure the provided path is correct (Hint: Use `pwd` in command line to verify current directory)")
        else:
            print(f"secrets.json file found at {secrets_path}. Beginning initialization of DataProcessor class.")
            self.secrets_path = secrets_path
        
        # Load secrets from the JSON file
        with open(self.secrets_path, 'r') as file:
            vals = json.load(file)
            self.binance_api_key = vals['BINANCE_API_KEY']
            self.binance_api_secret = vals['BINANCE_API_SECRET']
            self.frequency = vals['Trading Frequency (Yearly/Monthly/Weekly/Daily/Hourly/Minutely)']
            self.start_date = vals['Starting Date (YYYY-MM-DD)']
            self.end_date = vals['Ending Date (YYYY-MM-DD)']
            self.base_currency = vals['Base Currency']
            self.tickers = vals['Tickers of Interest']
            self.n = vals['Window Size']

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
            raise ValueError("Invalid frequency. Choose from 'Daily', 'Weekly', 'Monthly', or 'Yearly'.")

        # Convert tickers to trading pairs 
        self.tickers = [f"{ticker}{self.base_currency}" for ticker in self.tickers]
        
        # Fetch all the crypto data requested 
        self.crypto_data = {}
        self.crypto_live_data = {}
        # Find the path to the project root 
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        for ticker in tqdm(self.tickers, desc="Fetching Crypto Data", unit="pair",
            ncols=80, bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            colour="green", leave=True, dynamic_ncols=True):
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

            # Compute log daily returns column
            data['LogReturns'] = np.log(data['Close'] / data['Close'].shift(1))
            data.dropna(inplace=True)
            
            # Now use LogReturns instead of Close price for windowing
            data = data[['Open Time', 'LogReturns']]

            # Compute windowed returns column using log returns
            windowed_data = self.df_to_windowed_df(dataframe=data)
            if windowed_data is None:
                print(f"Warning: Could not create windowed data for {ticker}. Skipping.")
                continue
            data = windowed_data
            data.dropna(inplace=True)

            # Save the processed data to CSV
            data.to_csv(ticker_file_path, index=False)
            self.crypto_data[ticker] = data
            print(f"Data for {ticker} saved successfully to {ticker_file_path}")
        
        print("All crypto data fetched successfully.")
        # Make sure crypto_data isn't empty
        if not self.crypto_data:
            print("Warning: No crypto data was successfully processed.")
        
        print(f"Fetching the last {self.n+1} data points for each ticker to calculate {self.n} log returns.")
        for ticker in tqdm(self.tickers, desc="Fetching Live Data", unit="pair",
                    ncols=80, bar_format="{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                    colour="blue", leave=True, dynamic_ncols=True):
            try:
                # Get the most recent self.n+1 klines to calculate self.n log returns
                recent_klines = self.binance_client.get_klines(
                    symbol=ticker,
                    interval=interval,
                    limit=self.n+1
                )
            
                # Convert to numpy array with just the 'Close' price
                recent_closes = np.array([float(kline[4]) for kline in recent_klines])
                
                if len(recent_closes) >= 2:  # Need at least 2 prices to calculate 1 return
                    # Calculate log returns
                    recent_log_returns = np.log(recent_closes[1:] / recent_closes[:-1])
                    
                    if len(recent_log_returns) == self.n:
                        self.crypto_live_data[ticker] = recent_log_returns
                    else:
                        print(f"Warning: Could only fetch {len(recent_log_returns)} log returns for {ticker} instead of {self.n}.")
                        self.crypto_live_data[ticker] = recent_log_returns
                else:
                    print(f"Error: Not enough data points to calculate log returns for {ticker}.")
            except Exception as e:
                print(f"Error fetching live data for {ticker}: {e}")
                continue
        
        print("All live data fetched successfully.")

        # Computing autocorrelations and removing non-stationary data 
        # Import statsmodels acf function
        
        # Set threshold for significant autocorrelations
        # For 90% confidence interval, we use 1.645/sqrt(n)
        for ticker in tqdm(self.tickers, desc="Processing Autocorrelations", unit="pair"):
            # Fix: Check if ticker exists in crypto_data and if 'Target' column exists
            if (ticker not in self.crypto_data or 
                self.crypto_data[ticker] is None or 
                self.crypto_data[ticker].empty or 
                'Target' not in self.crypto_data[ticker].columns):
                print(f"Ticker {ticker} missing required data. Skipping autocorrelation threshold calculation.")
                continue  # Skip this ticker if 'Target' column is missing

            self.ACF_THRESHOLD = 1.645 / np.sqrt(len(self.crypto_data[ticker]['Target'])) * 1.3
            print(f"Ticker: {ticker} - Autocorrelation significance threshold: {self.ACF_THRESHOLD:.4f}")
            print(f"Allowed number of deviations: {np.ceil(self.n * 0.1 + 1)}")
            self.compute_autocorrelations()

            # Using both acf methods for each ticker
            for tkr, custom_acf in self.acfs.items():
                custom_acf = np.array(custom_acf)
                # Get statsmodels acf for the same data
                sm_acf = statsmodels_acf(self.crypto_data[tkr]['Target'].values, nlags=self.n)
                
                # Count how many autocorrelations are above the threshold for each method
                custom_count_above = np.sum(np.abs(custom_acf) > self.ACF_THRESHOLD)
                sm_count_above = np.sum(np.abs(sm_acf[1:]) > self.ACF_THRESHOLD)  # Skip lag 0
                
                # Calculate maximum allowed deviations for 90% confidence
                max_allowed_deviations = np.ceil(self.n * 0.1 + 3)
                
                # Only remove if both methods agree on non-stationarity
                if custom_count_above > max_allowed_deviations and sm_count_above > max_allowed_deviations:
                    print(f"Ticker: {tkr} REMOVED (Non-Stationarity Detected by both methods)")
                    print(f"  - Custom ACF: {custom_count_above} autocorrelations above threshold")
                    print(f"  - Statsmodels ACF: {sm_count_above} autocorrelations above threshold")
                    try:
                        os.remove(os.path.join(data_dir, f"{tkr}.csv"))
                        self.crypto_data.pop(tkr, None)
                        self.crypto_live_data.pop(tkr, None)
                        print(f"Data for {tkr} removed successfully.")
                    except Exception as e:
                        print(f"Error removing data for {tkr}: {e}")
                else:
                    print(f"Ticker: {tkr} PASSED (Stationarity Detected)")
                    print(f"  - Custom ACF: {custom_count_above} autocorrelations above threshold")
                    print(f"  - Statsmodels ACF: {sm_count_above} autocorrelations above threshold")

        print("Autocorrelation analysis completed successfully.")
        print("Data processing completed successfully.")
        
    # Convert string date to datetime object
    def str_to_datetime(self, date_str: str) -> datetime:
        return datetime.strptime(date_str, '%Y-%m-%d')
        
    # Compute windowed returns column 
    def df_to_windowed_df(self, dataframe: pd.DataFrame):
        first_date = self.str_to_datetime(self.start_date)
        last_date  = self.str_to_datetime(self.end_date)

        dataframe = dataframe.set_index('Open Time')
        sorted_dates = sorted(dataframe.index)
        if len(sorted_dates) <= self.n:
            print(f"Error: Not enough data points ({len(sorted_dates)}) for window size {self.n}")
            return None

        target_date_index = self.n  # Start where we have n previous points
        dates = []
        X, Y = [], []

        while target_date_index < len(sorted_dates):
            target_date = sorted_dates[target_date_index]

            df_subset = dataframe.loc[:target_date].tail(self.n+1)
            if len(df_subset) != self.n+1:
                print(f'Error: Window of size {self.n} is too large for date {target_date}')
                target_date_index += 1
                continue

            # Use LogReturns instead of Close prices
            values = df_subset['LogReturns'].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

            # Update target_date_index to process the next date
            target_date_index += 1

        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates
        X = np.array(X)
        for i in range(self.n):
            # Now these are log return lags instead of price lags
            ret_df[f'LogReturn-Lag-{self.n - i}'] = X[:, i]
        ret_df['Target'] = Y  # This is now the target log return

        self.windowed_df = ret_df

        return self.windowed_df
    
    def compute_autocorrelations(self):
        self.acfs = {}
        # Compute the autocorrelations of the log returns for each ticker 
        for ticker, data in self.crypto_data.items():
            # Compute autocorrelations
            acf = qcs.compute_autocorrelations(data['Target'].values, self.n)
            self.acfs[ticker] = acf
        return 1

    
if __name__ == "__main__":
    dp = DataProcessor()  # One-Liner is all it takes to initialize the DataProcessor class)
    # For example, let's print "BTCUSDT" data
    print(dp.crypto_data['BTCUSDT'].head())  # Print the first 5 rows of BTCUSDT data
    print(dp.crypto_live_data['BTCUSDT'])  # Print the live data for predictions