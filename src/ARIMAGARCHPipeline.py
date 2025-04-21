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