# CryptoXQuant-Alpha (Project CxQ-A)
A multi-faceted research project into a complex alpha incorporating and combining the usage of classical time series models, LSTMs and statistical learning techniques to find a dynamic alpha to optimize a Cryptocurrency portfolio. 

**Note: This project is purely for interest and research purposes. I am not responsible for any losses that occur as a result of using this GitHub Repository, please use at your own risk.**

## Brief Description
Multi-Disciplinary Quantitative Trading Strategy invoking the usage of:
- Classic Time Series Modeling (ARIMA-GARCH Joint Model) `C++` with `PyBind11` for efficiency
- Long-Short-Term-Memory (LSTM) Model (Form of Recurrent Neural Network (RNN)) `tensorflow` with `metal` support (MacOS X)
- Classical Statistical Learning Techniques
- Computation of covariance between assets
- Combining multi-faceted strategy into single alpha which will be used in parrallel with the asset relations to compute the most optimal low-risk high-return (efficient frontier) portfolios. 

## Usage
### Hidden Files
To be able to fetch Cryptocurrency Historical Data, you are required to provide a Binance API Key and a Secret Key. These are available for free by simply switching your Binance account to a PRO account and retrieving your keys from the Binance API Page. 

Then, you need to set up the `secrets.json` file which will be located at `$PROJ_ROOT/secrets/secrets.json`. The base file structure is shown below:

`secrets.json`

```json
{
    "BINANCE_API_KEY": "Enter Binance API Key Here", 
    "BINANCE_API_SECRET": "Enter Binance Secret Key Here",
    "Trading Frequency (Yearly/Monthly/Weekly/Daily/Hourly/Minutely)": "Daily",
    "Starting Date (YYYY-MM-DD)": "2015-01-01", 
    "Ending Date (YYYY-MM-DD)": "2025-04-01",
    "Base Currency": "USDT", 
    "Tickers of Interest": [
        "BTC", "ETH", "XRP", "SOL", "LTC", "DOGE", "BNB", 
        "ADA", "TRX", "LINK", "XLM", "FIL", "MATIC", "DOT",
        "ALGO", "ICP", "BCH", "UNI", "ETC", "VET", "EOS", 
        "AAVE", "NEAR", "ATOM", "FTM", "THETA", "XTZ", 
        "ZEC", "MANA", "SAND", "BAT", "CHZ", "HBAR", "LDO",
        "CRV", "KSM", "XEM", "ZRX", "QTUM", "DGB", "WAVES",
        "HNT", "XVG", "DASH", "ZIL", "NANO", "OMG", "REN",
        "1INCH", "SUSHI", "YFI", "COMP", "SNX", "LRC", "STMX",
        "FET", "STPT", "LEND", "BAND", "ENJ", "LPT", "RLC"
    ],
    "Window Size": 40
}
```
After this has ben set up, calling the DataProcessor Class initializer will automatically use the data you have provided in the secrets.json file to populate and compute the models for usage. I would recommend against using Hourly Data (Very noisy and remains non-stationary even after differencing operator applied) and Minutely Data (Same issue as Hourly Data but with additional extremely long runtime for fetching data using Binance API). 

## Documentation 
If you are interested in the Quantitative aspect of any part of the strategies, I have uploaded my notes of each section in the `docs` sections of the repository. Alternatively the hyperlinks below will be updated when documentation is available for it:

[Project Overview](docs/overview.md)

[Stationarity Checks (ADF and KPSS)](docs/stationarity_checks.md)
