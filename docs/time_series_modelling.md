# Time Series Modelling (ARIMA-GARCH Model)
Now that our data has been processed and we have a stationary dataset for each cryptocurrency (via the ADF and KPSS tests), we can move to our first modelling strategy which is to fit the time series data to an ARIMA-GARCH model. The ARIMA-GARCH model was chosen here because it works well when our returns are linear and homoskedastic (which we made them to be) but further than that it alows us to capture time-varying risk so our model is able to forecast both the expected return and the expected volatility.

We formula the model as follows:

1) **Mean (ARIMA($p, d, q$))**: Since our data is already stationary, we set $d=0$ (one less parameter to optimize) and work directly with, $$r_t = \mu + \sum_{i = 1}^p {\phi_i} r_{t - i} + \sum_{j = 1}^q \theta_j \epsilon_{t - j} + \epsilon_t$$

2) **Variance (GARCH($r, s$))**: We model the time varying volatility $\sigma_t^2$ of $\epsilon_t$ with, $$\epsilon_t = \sigma_t z_t, \;\; z_t \sim N(0, 1), \;\; \sigma_t^2 = \omega \sum_{l = 1}^{r} \alpha_l \epsilon_{t - l}^2 \sum_{m = 1}^s \beta_m \sigma_{t - m}^2$$

Now, we compute the best estimations for the parameters jointly, i.e. $\Theta = \{\mu, \phi_1, \dots, \phi_p, \theta_1, \dots, \theta_q; \omega, \alpha_1, \dots, \alpha_r, \beta_1, \dots, \beta_s\}$ via a maximum likelihood computation of $$\mathcal{L}(\Theta)
= \sum_{t=1}^T \Bigl[-\tfrac12\Bigl(\ln(2\pi) + \ln\sigma_t^2 + \frac{\varepsilon_t^2}{\sigma_t^2}\Bigr)\Bigr]$$

Now, the second part of this optimization is for the order of the model, i.e. the values of $(p, q, r, s)$. In order to do this effectively, we use the Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC) on a pure ARIMA model to choose $(p, q)$ and then fit a GARCH model on the ARIMA residuals and choose $(r, s)$ again through the use of either AIC or BIC. 

There will also be a file in `experiments` where the effectiveness of the ARIMA-GARCH model will be tested on a couple of cryptocurrencies before deployment to all of them within the `secrets.json` file. 

Further to this, in order to have some guidance regarding its predictive power, I have decided to take inspiration from the LSTM portion of this project and split the time series dataset prematurely into train/validation/test splits just as we would for the LSTM model. I have decided on a split of 70%/15%/15% which will be allocated in the following way:

| **Split**     | **Portion** | **Use**                          |
|---------------|-------------|----------------------------------|
| Train         | 70%         | Fit ARIMA‑GARCH Parameters       |
| Validation    | 15%         | Tune $(p, q, r, s)$              |
| Test          | 15%         | Estimating Predictive Power      |

The relevant file for the experimentation is: [ARIMA-GARCH Modelling](../experiments/Arima_Garch_Modelling.ipynb)

Additionally, it is worth noting, to avoid the same model having to be built over and over, a directory called `timeseriesmodels` should be made/is made which will host any trained model parameters from the ARIMA-GARCH models which can instantly be reconstructed. 

The relevant code is:
```python
# ARIMAGARCHPipeline.py
import os
import json
import pickle
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from arch import arch_model

class ARIMAGARCHPipeline:
    def __init__(self,
                 df: pd.DataFrame,
                 model_path: str = None,
                 # only used if model_path is None
                 train_frac: float = 0.70,
                 val_frac:   float = 0.15,
                 max_p: int = 5, max_q: int = 5,
                 garch_ps: tuple = (1,2), garch_qs: tuple = (1,2)
    ):
        """
        If model_path is provided, attempts to load saved orders and pickled models.
        Otherwise, prepares splits for tuning.
        """
        # 1) Chronological split
        df = df.sort_values('Open Time').reset_index(drop=True)
        T = len(df)
        te = int(train_frac * T)
        ve = int((train_frac + val_frac) * T)
        self.train = df.iloc[:te]
        self.val   = df.iloc[te:ve]
        self.test  = df.iloc[ve:]

        # placeholders for orders & models
        self.p = self.q = None
        self.r = self.s = None
        self.arima_model = None
        self.garch_model = None
        self.metrics = {}

        # grid‐search settings
        self.max_p = max_p
        self.max_q = max_q
        self.garch_ps = garch_ps
        self.garch_qs = garch_qs

        # load existing or wait for tuning
        if model_path:
            self._load_and_fit(model_path)

    def _load_and_fit(self, model_path: str):
        """
        Load orders from JSON. If a pickle of fitted models exists, unpickle them.
        Otherwise fit ARIMA and GARCH once and then pickle for next time.
        """
        # read orders
        with open(model_path, 'r') as f:
            info = json.load(f)
        self.p, self.q = info['arima_order']
        self.r, self.s = info['garch_order']

        # check for pickle
        pkl_path = model_path.replace('.txt', '.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as pf:
                models = pickle.load(pf)
            self.arima_model = models['arima']
            self.garch_model = models['garch']
            return

        # no pickle: fit once on train+val
        combined = pd.concat([self.train, self.val], ignore_index=True)
        r_combined = combined['Covariance-LogDiff']

        # ARIMA fit
        self.arima_model = auto_arima(
            r_combined,
            order=(self.p, 0, self.q),
            stationary=True,
            suppress_warnings=True
        )
        resid = pd.Series(self.arima_model.resid())

        # GARCH fit
        self.garch_model = arch_model(
            resid,
            vol='Garch', p=self.r, q=self.s,
            dist='t', rescale=False
        ).fit(disp='off')

        # pickle for future
        with open(pkl_path, 'wb') as pf:
            pickle.dump({
                'arima': self.arima_model,
                'garch': self.garch_model
            }, pf)

    def tune_and_fit(self):
        """
        1) Select ARIMA(p,0,q) on train via AIC.
        2) Refit ARIMA on train+val.
        3) Grid‐search GARCH(r,s) on residuals of combined.
        4) Refit chosen GARCH on combined.
        """
        # ARIMA order search on train
        r_train = self.train['Covariance-LogDiff']
        arima_init = auto_arima(
            r_train,
            d=0, stationary=True,
            max_p=self.max_p, max_q=self.max_q,
            information_criterion='aic',
            suppress_warnings=True,
            stepwise=True
        )
        self.p, self.q = arima_init.order[0], arima_init.order[2]

        # Refit ARIMA on train+val
        combined = pd.concat([self.train, self.val], ignore_index=True)
        r_combined = combined['Covariance-LogDiff']
        self.arima_model = auto_arima(
            r_combined,
            order=(self.p, 0, self.q),
            stationary=True,
            suppress_warnings=True
        )

        # Residuals for GARCH
        resid = pd.Series(self.arima_model.resid())

        # GARCH grid‐search on combined residuals
        best_aic = np.inf
        best_rs = (None, None)
        for r in self.garch_ps:
            for s in self.garch_qs:
                try:
                    test_garch = arch_model(
                        resid,
                        vol='Garch', p=r, q=s,
                        dist='t', rescale=False
                    ).fit(disp='off')
                    if test_garch.aic < best_aic:
                        best_aic = test_garch.aic
                        best_rs = (r, s)
                except Exception:
                    continue

        self.r, self.s = best_rs

        # Final GARCH fit
        self.garch_model = arch_model(
            resid,
            vol='Garch', p=self.r, q=self.s,
            dist='t', rescale=False
        ).fit(disp='off')

        return {
            'arima_order': (self.p, 0, self.q),
            'garch_order': (self.r, self.s),
            'arima_aic': self.arima_model.aic(),
            'garch_aic': self.garch_model.aic
        }

    def evaluate(self):
        """
        One‐shot forecasts on test split, computing MSE and 1σ coverage.
        """
        r_test = self.test['Covariance-LogDiff'].reset_index(drop=True)
        n_test = len(r_test)

        # ARIMA mean forecasts
        mu_pred = self.arima_model.predict(n_periods=n_test)

        # GARCH variance forecasts
        fc = self.garch_model.forecast(horizon=n_test, reindex=False)
        var_pred = fc.variance.values.flatten()
        sigma_pred = np.sqrt(var_pred)

        # Metrics
        errors = (r_test.values - mu_pred) ** 2
        mse = errors.mean()
        coverage = np.mean(np.abs(r_test.values - mu_pred) <= sigma_pred)

        self.metrics = {
            'test_mse': mse,
            '1sigma_coverage': coverage
        }
        return self.metrics

    def save_model(self, model_path: str):
        """
        Save orders to JSON and pickle the fitted ARIMA & GARCH models.
        """
        # JSON for orders
        info = {
            'arima_order': [self.p, self.q],
            'garch_order': [self.r, self.s]
        }
        with open(model_path, 'w') as f:
            json.dump(info, f)

        # Pickle for model objects
        pkl_path = model_path.replace('.txt', '.pkl')
        with open(pkl_path, 'wb') as pf:
            pickle.dump({
                'arima': self.arima_model,
                'garch': self.garch_model
            }, pf)

    def forecast_next(self, live_series: np.ndarray):
        """
        One‐step‐ahead forecast on new live data:
        returns (mean_return, volatility).
        """
        # ARIMA mean
        arima_fc = self.arima_model.predict(n_periods=1)
        mu = arima_fc.iloc[0] if hasattr(arima_fc, 'iloc') else arima_fc[0]

        # GARCH vol
        fc = self.garch_model.forecast(horizon=1, reindex=False)
        var = fc.variance.values.flatten()[0]
        sigma = np.sqrt(var)

        return mu, sigma

    def run(self):
        """
        Convenience: tune + fit + evaluate → returns combined info dict.
        """
        info = self.tune_and_fit()
        metrics = self.evaluate()
        return {**info, **metrics}
```