# Crypto_TS_Modeling 
A research project for ARIMA-GARCH Model Fitting alongside computation of a covariance matrix between numerous assets in order to optimize a portfolio for multiple trading frequencies. 

**Note: This project is purely for interest and research purposes. I am not responsible for any losses that occur as a result of using this GitHub Repository, please use at your own risk.**

## Brief Description
Multi-Disciplinary Quantitative Trading Strategy invoking the usage of:
- Manipulation of Time Series Data to ensure stationarity using ADF and KPSS Tests. 
- Classic Time Series Modeling (ARIMA-GARCH Joint Model) `C++` with `PyBind11` for efficiency
- Computation of covariance between assets
- Combining multi-faceted strategy into single alpha which will be used in parallel with the asset relations to compute the most optimal low-risk high-return (efficient frontier) portfolios. 

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
    ]
}
```
After this has ben set up, calling the DataProcessor Class initializer will automatically use the data you have provided in the secrets.json file to populate and compute the models for usage. I would recommend against using Hourly Data (Very noisy and remains non-stationary even after differencing operator applied) and Minutely Data (Same issue as Hourly Data but with additional extremely long runtime for fetching data using Binance API). 

Now, to configure your environment so everything works smoothly, 
```shell 
conda env create -f environment.yml
conda activate Crypto_TS_Modeling
```

Finally, a fully automated procedure is now ready for trading information using, 
```shell
python main.py
```

**Note: If an error shows like below, just run the script again.** 
```shell
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 2606, in pandas._libs.hashtable.Int64HashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 2630, in pandas._libs.hashtable.Int64HashTable.get_item
KeyError: 0
```

## Documentation 
If you are interested in the technical aspect of any part of the strategies, I have uploaded my notes of each section in the `docs` sections of the repository. Alternatively the hyperlinks below will be updated when documentation is available for it:

[Project Overview](docs/overview.md)

[Stationarity Checks (ADF and KPSS)](docs/stationarity_checks.md)

[Technical Indicators & Lags](docs/technical_indicators_lags.md)

[Covariance Computation](docs/covariance_computation.md)

[ARIMA-GARCH Time Series Modelling](docs/time_series_modelling.md)
