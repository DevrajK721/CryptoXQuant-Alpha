from src import DataProcessor as DP

dp = DP.DataProcessor() # This one-liner will read from secrets.json and download all the ticker data. 

# Check the tickers have been initialized correctly
print(f"Tickers: {dp.tickers}")

# Check the historical close price for QTUMUSDT 
print(f"Historical close price for QTUMUSDT: {dp.crypto_data['QTUMUSDT']['Close'].head()}")

# Check the live price for ETHUSDT 
print(f"Live price for ETHUSDT: {dp.crypto_live_data['ETHUSDT']}")