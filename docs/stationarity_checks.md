# Stationarity Checks and Procedure 

As inputs into time series models, LSTMs and standard statistical learning models, it is important to ensure that the time series data is stationary. As such, three separate mechanisms (checks) are used to confirm the validity of the input data:
- Augmented Dickey-Fuller Test (ADF)
    - If ADF $p$-value $\geq 0.05$ - Still Unit Root (Non-Stationary)
- Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    - If KPSS $p$-value $\leq 0.05$ - Still Non-Stationary
- Autocorrelation Function/Partial Autocorrelation Function (ACF/PACF) Analysis
    - Useful when considering visually, not used in this instance 

To produce the best models possible, we will only be considering data once it passes both the ADF test and the KPSS test within that 5\% confidence bound. 

Further to that, to ensure the ability to recover equivalent predictions, the data for what variable transformations have occurred will be stored in addition to a ticker's data. For clarity, this would mean for example, if `ETHUSDT` time series data required the transformation `np.log(data).diff().diff()` (Logging followed by two differencing operators) in order to become stationary, this data would be stored under `transforms['ETHUSDT'] = ['log', 'diff', 'diff']` so that a prediction from another dataset for price for example would have its equivalent able to be computed through the reverse of these operations. 

The relevant functions are shown below for clarity and for visualization of how the above has been ensured.

```python 
# DataProcessor.py
...
def is_stationary(self, s, signif: float = 0.05):
    # ADF Test
    adf_p = adfuller(s.dropna())[1]

    # KPSS Test 
    kpss_p= kpss  (s.dropna(), regression='c')[1]

    return (adf_p < signif) and (kpss_p > signif)

def make_stationary(self, max_diff: int = 5):
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
...
```

