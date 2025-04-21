# main.py
import os, json, warnings
import pandas as pd
from src import DataProcessor as DP
import covcomp
from src.ARIMAGARCHPipeline import ARIMAGARCHPipeline as AGP

warnings.filterwarnings("ignore")

# 1) init data
dp = DP.DataProcessor()

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

# ─── 5) Simple signal logic ────────────────────────────────────────────────────
# set your cost hurdle (e.g. 0.2% round-trip) and S/N threshold
c_min = 0.0001   # 0.1%
k     = 0.3     # require at least a 1σ signal

print("\n=== TRADING SIGNALS ===")
for tk, f in forecasts.items():
    mu    = f['mean']
    sigma = f['sigma']
    signal = mu / sigma if sigma>0 else 0.0

    if (mu > c_min) and (signal > k):
        action = "BUY"
    elif (mu < -c_min) and (signal < -k):
        action = "SELL"
    else:
        action = "HOLD"

    print(f"{tk}: μ={mu:.4f}, σ={sigma:.4f}, S/N={signal:.2f} → {action}")