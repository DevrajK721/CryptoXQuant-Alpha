# Technical Indicators & Lags 
As part of the LSTM & Statistical Learning models (NOT Time Series Models), we can use popular technical indicators and $k$-lags for assistance in the prediction process. 

The list of indicators being added (mainly by popularity) are:
- Exponential Moving Average (EMA) $\{10, 20, 50\}$
- Relative Strength Index (RSI) $\{14\}$
- Bollinger Bands (BB) $\{2
\theta, 2
\sigma\}$
- Rate of Change (ROC) $\{10\}$
- Rolling Volatility (RV) $\{20\}$
- Moving-Average-Convergence-Divergence (MACD) $\{12, 26\}$
- Lags ($k$) $\{1, 2, \dots, 30\}$

These indicators have been computed as in the following code snippet:
```python 
def compute_technical_indicators_and_lags(self, df: pd.DataFrame) -> pd.DataFrame:        
    # Compute EMA Exponential Moving Averages (EMA's) 
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 2) Relative Strength Index (14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # 3) Bollinger Bands (20, ±2σ)
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = rolling_mean + 2 * rolling_std
    df['BB_lower'] = rolling_mean - 2 * rolling_std
    
    # 4) Rate of Change (10)
    df['ROC_10'] = df['Close'].pct_change(periods=10)
    
    # 5) Rolling Volatility (20-period std of returns)
    df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
    
    # 6) MACD (12,26) and signal (9)
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 7) Lags 1 through 30
    for lag in range(1, 31):
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    
    return df
```

It is important to note that these indicators will not all necessarily be used in conjunction with one another. There will be a tactical implementation of choosing these indicators for each model through incremental additions and their effect on the predictive power of the model via walk-forward or cross-validation. Overall, in each model there is an aim to:
- Select a small set of diverse indicators covering trend, momentum and volatility
- Avoid the use of collinear indicators, e.g. the 10-Day and 20-Day EMA 
- Integrate smoothly into LSTMs and Statistical Models using thresholds showing positive and negative indication, e.g. RSI > 70 = Good.

