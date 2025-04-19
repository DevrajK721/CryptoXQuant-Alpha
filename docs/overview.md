# Project Overview 

The fundamental idea behind this project is to do two things:
1) Build an LSTM and high quality statistical time series model for a fixed amount of assets downloaded from the Binance API to successfully forecast returns 
2) Use the historical data to also carry out portfolio optimization for lowest risk matched with highest return, aka the so-called efficient frontier.
3) *Optional: Use classical machine learning techniques as an additional attempt to predict future prices*
4) Combining all of these factors to have a robust, low error and high return portfolio optimization tool to be used on a daily basis. 

This project aims to be inclusive of many features of Quantitative Finance and FinTech including:
- Time Series Models 
- Working with large datasets efficiently 
- Writing C++ code with PyBind11 to ensure the highest amount of efficiency with still extremely easy to write code 
- Machine Learning to build LSTM's (Deep Learning) using TensorFlow 
- Backtesting trading strategies     

Overall, this project should be a very good insight into fundamental quantitative analysis of an asset (Crypto in this case but generalizable) and a good way to practice real-world problem solving. As a top level overview, this project entails the following tasks which will later be broken up into multiple sub-tasks.

- [x] Use the Binance API to fetch historical data for a list of Cryptocurrencies we would be interested in making a portfolio with. 
- [x] Compute additional parameters from the fetched historical data, e.g. lags, technical indicators (EMA, RSI, etc.)
- [x] Post-Processing the historical data to make sure it is suitable for building LSTM's, statistical time series models, etc. by e.g. checking ACF, PACF, etc. 
- [ ] Using the historical data to build/compute:
	- [ ] Correlations/Covariances between the assets
	- [ ] Statistical Time-Series Models, e.g. ARIMA-GARCH 
	- [ ] Deep Learning LSTM's using Tensorflow 
	- [ ] Classical Machine Learning Models, e.g. Linear Regression 
- [ ] Efficiently combining the outputs/predictions from all the above sources to have a robust, low risk and high return portfolio optimizer
- [ ] Backtesting this "strategy" using basic P/L data from the historical dataset 
- [ ] Iterating upon the design to obtain good results from backtesting (effectively the validation set)
- [ ] Porting the process to QuantConnect or a live backtesting session using personal Binance account to share real world results 
- [ ] Organizing the overall project structure to be clean, easy to read and well documented with consistent Git activity showing good use of version control. 

## Tasks Overview & Considerations 
### Fetching Historical Data via Binance API 
- Use `python-binance` library. 
- Handle rate limits when looking into live execution. 
- Potentially consider using a time-series database to avoid re-downloads or have code implementation to avoid this. 
- Log only the data you need, most likely `Ticker`, `Timestamp` and `Close Price` 

### Post-Processing, Stationarity Checks & Technical Indicators 
- Use pandas to difference/log returns 
- Confirm stationarity using Augmented Dickey-Fuller and KPSS tests and flag non-stationary data for differencing 
- Use ACF/PACF analysis to help with model selection. 
- Choose indicators tactically, having too many can get confusing, each indicator should have its incremental predictive power tested via walk-forward or cross validation. 
	- Select a small set of diverse indicators covering trend, momentum and volatility. 
	- Avoid using collinear indicators, e.g. 10-Day MA and 20-Day MA 
	- Integrate into LSTM and Classical Statistical Models (Not Time-Series Models) using simple threshold rules, e.g. RSI > 70 -> long position open 
### Computing Correlations & Covariance Estimations 
- Use shrinkage estimators (`Scikit-Learn` - `LedoitWolf` estimator)
- This will help balance the sample covariance with a structured target. 
### Statistical Time-Series Models (ARIMA-GARCH)
- Write C++ code piped with PyBind11 for ease of use in Python to automate the `p, d, q` grid search for the best potential model 
	- Alternatively, use `pmdarima.auto_arima` and `arch` package. 
	- Leverage parallelism using `joblib` to fit multiple specs concurrently. 
### Deep Learning LSTMs 
- Beyond just using raw returns and lags, be sure to incorporate the technical indicators found to have the biggest effect. 
- Stack LSTM/GRU layers, tune learning rate, sequence length and batch size. 
- Monitor overfitting with early stopping and dropout. 
- Use expanding-window walk-forward splits to mimic real-time forecasting.  
### Classical Machine Learning Models 
- Normalize of standardize inputs and include lagged features and technical indicators. 
- Use out-of-fold predictions to train a meta-learner which will reduce single-model bias. 
- Use multiple regressors, e.g. Linear, Ridge, Random Forest, XGBoost and stack their predictions 
### Combining Predictions for Portfolio Optimization 
- Use ensemble weights based on predictive performance (establish robust criteria for this) and incorporate Bayesian Model Averaging or Shrinkage Blending. 
- Dynamically adjust the weights using an optimizer for the Sharpe ratio (form the efficient Frontier)
- Solve mean-variance with the expected returns from the ensembles and shrinking covariance 
- Add constraints (max drawdown, turnover limits) and regularization (L2 Norm on the weights)
### Backtesting the Strategy 
- Either employ a dedicated backtester, e.g `Zipline` or `backtrader` if it integrates easily enough or use a simple P/L from the remaining dataset. 
- Track the drawdown, Sharpe, Calmar and turnover. 
- Generate equity curves and analytics for each run to keep track of strategy improvement. 
- Add in stop loss and take profits if strategies prove to be quite risky. 
### Iteration & Validation 
- Implement automated hyperparameter tuning (Optuna), track experiments with MLflow and conduct out-of-sample tests. 
- Log parameters, metrics and results to properly test different strategies 
### Live Deployment 
- Use Binance TestNet or QuantConnect or even Binance Live Trading account if confident in strategy to live test. 
- Monitor real-world effects on P&L 
### Organizing Project Structure & Documentation 
- Adopt a clear repo layout, use Black Formatter to keep everything organized and write README and Sphinx docs. 
- For the repo layout splits into:
	- `data`
	- `src`
	- `tests`
	- `notebooks`
## Estimated Timeframe
The timeframe for this project is purely discretionary but should give a good indication for how things should be moving. 

**Total Work: ~120-150 Hours**
**Weekly Commitment: ~6-11 Hours**
**Estimated Duration: ~3-5 Months**

### Task Breakdown - Timing 
| Phase                                      | Est. Hours | Est. Weeks (8 h/week) |
| ------------------------------------------ | ---------: | --------------------: |
| **1. Setup & Data Ingestion**              |       *15 h* |                  *2 wk* |
| **2. Preprocessing & Stationarity Checks** |       *12 h* |                *1.5 wk* |
| **3. Correlation & Covariance**            |       *10 h* |               *1.25 wk* |
| **4. ARIMA‑GARCH Modeling**                |       *20 h* |                *2.5 wk* |
| **5. LSTM Prototyping**                    |       *25 h* |                *3.1 wk* |
| **6. Classical ML Models**                 |       *12 h* |                *1.5 wk* |
| **7. Ensemble & Portfolio Optimization**   |       *15 h* |                *1.9 wk* |
| **8. Backtesting Framework**               |       *12 h* |                *1.5 wk* |
| **9. Iteration & Validation**              |       *12 h* |                *1.5 wk* |
| **10. Deployment & Monitoring**            |       *15 h* |                *1.9 wk* |
| **11. Documentation & CI/CD**              |       *10 h* |               *1.25 wk* |
| **Buffer & Polish**                        |       *10 h* |               *1.25 wk* |
| **Total**                                  |  **148 h** |        **18.5 weeks** |

