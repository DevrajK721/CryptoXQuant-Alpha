# main.py
import os, json, warnings
import pandas as pd
from src import DataProcessor as DP
import covcomp
import time
from src.ARIMAGARCHPipeline import ARIMAGARCHPipeline as AGP
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np

warnings.filterwarnings("ignore")

# 1) init data
dp = DP.DataProcessor(secrets_path='secrets/secrets.json')

# 2) covariances (unchanged)
cov_input = {
    t: dp.crypto_data[t][['Open Time','Covariance-LogDiff']]
    for t in dp.tickers
}
cov_dict = covcomp.compute_covariance(cov_input, shrinkage=0.05)
cov_df   = pd.DataFrame.from_dict(
    {t: dict(cov_dict[t]) for t in cov_dict}, orient='index'
)
print("Covariance matrix:\n", cov_df)

# ensure folder
model_dir = 'timeseriesmodels'
os.makedirs(model_dir, exist_ok=True)

# 3) fit each model, save, and forecast next
results = {}
forecasts = {}

for ticker, df in cov_input.items():
    print(f"\n** Processing {ticker} **")
    model_path = os.path.join(model_dir, f"{ticker}.txt")
    
    if os.path.exists(model_path):
        # load & fit from saved orders
        pipeline = AGP(df, model_path=model_path)
        print(" Loaded existing model orders.")
    else:
        # tune & fit from scratch
        pipeline = AGP(df)
        out = pipeline.run()
        pipeline.save_model(model_path)
        print(" Fitted & saved new model:", out['arima_order'], out['garch_order'])
    
    # evaluate & collect
    out = pipeline.metrics or pipeline.evaluate()
    results[ticker] = out
    
    # forecast next using live data
    live = dp.crypto_live_data[ticker]  # array of last n stationary returns
    mu, sigma = pipeline.forecast_next(live)
    forecasts[ticker] = {'mean': mu, 'sigma': sigma}
    print(f" Next-step forecast → mean: {mu:.5f}, vol: {sigma:.5f}")

# 4) summary
print("\n=== BACKTEST RESULTS ===")
for tk, r in results.items():
    print(f"{tk}: MSE={r.get('test_mse'):.6f}, 1σ cov={r.get('1sigma_coverage'):.2%}")

print("\n=== LIVE FORECASTS ===")
for tk, f in forecasts.items():
    print(f"{tk}: μ={f['mean']:.5f}, σ={f['sigma']:.5f}")

# ─── 5) Portfolio Optimization for Next 24 h with P(return>0) ≥ 70% ─────────
# 5.1) build expected‐return vector μ and covariance matrix Σ
tickers = list(forecasts.keys())
mu_vec  = np.array([forecasts[t]['mean']  for t in tickers])
cov_mat = cov_df.loc[tickers, tickers].values

# 5.2) get the z‐score corresponding to x% probability
x = 59 # Enter Percent Risk 
z_target = norm.ppf(x / 100)   

# 5.3) define objective = –Sharpe ratio
def neg_sharpe(w, mu, cov):
    ret = w @ mu
    vol = np.sqrt(w @ cov @ w)
    return -ret/vol

# 5.4) constraints:
cons = [
    # weights sum to 1
    {'type':'eq', 'fun': lambda w: np.sum(w) - 1},
    # enforce w·μ - z_target * sqrt(w·Σ·w) ≥ 0
    {'type':'ineq',
     'fun': lambda w, mu=mu_vec, cov=cov_mat, z=z_target:
         (w @ mu) - z * np.sqrt(w @ cov @ w)
    }
]

# long‐only bounds
bnds = tuple((0.0, 1.0) for _ in tickers)

# 5.5) initial guess: equal‐weight
x0 = np.ones(len(tickers)) / len(tickers)

# 5.6) solve
res = minimize(
    neg_sharpe, x0,
    args=(mu_vec, cov_mat),
    method='SLSQP',
    bounds=bnds,
    constraints=cons
)

if not res.success:
    raise ValueError("Optimization failed: " + res.message)

w_opt = res.x

# 5.7) compute portfolio metrics
port_ret    = w_opt @ mu_vec
port_vol    = np.sqrt(w_opt @ cov_mat @ w_opt)
port_sharpe = port_ret / port_vol
prob_pos    = norm.cdf(port_sharpe)

# 5.8) display
print("\n=== OPTIMAL PORTFOLIO FOR NEXT 24 H (P>0 ≥ 70%) ===")
for t, w in zip(tickers, w_opt):
    if w > 1e-3:  # show only >0.1%
        print(f"  {t}: {w*100:5.2f}%")
        
print(f"\n Expected return:     {port_ret*100:5.2f}%")
print(f" Expected volatility: {port_vol*100:5.2f}%")
print(f" Sharpe ratio:        {port_sharpe:5.2f}")
print(f" P(return>0):         {prob_pos*100:5.1f}%")