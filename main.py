import pandas as pd
from src import DataProcessor as DP
import covcomp

# 1) Initialize and load all your processed data
dp = DP.DataProcessor()

# 2) Build the “input dict” that covcomp expects:
#    key = ticker, value = DataFrame with exactly ['Open Time','Covariance-LogDiff']
cov_input = {
    t: dp.crypto_data[t][['Open Time','Covariance-LogDiff']]
    for t in dp.tickers
}

# 3) Compute the shrinkage‑adjusted covariance matrix:
#    You can tune `shrinkage` between 0 (no shrink) and 1 (full identity).
cov_dict = covcomp.compute_covariance(cov_input, shrinkage=0.05)

# 4) Turn it into a friendly DataFrame
cov_df = pd.DataFrame.from_dict(
    { t: dict(cov_dict[t]) for t in cov_dict }, 
    orient='index'
)
print("Shrinkage‑adjusted covariance matrix:\n", cov_df)