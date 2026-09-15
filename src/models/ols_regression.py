"""Ordinary Least Squares (OLS / MCO) Regression with complete econometric diagnostics."""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from src.models.base_model import BaseRegressionModel


class OLSRegressionModel(BaseRegressionModel):
    """Ordinary Least Squares (MCO) with statistical assumptions & diagnostic validation."""

    def __init__(self, model_name: str = "Regresion Lineal MCO (OLS)"):
        super().__init__(model_name=model_name)
        self.summary_table: str = ""
        self.coefficients_: pd.Series = pd.Series(dtype=float)
        self.pvalues_: pd.Series = pd.Series(dtype=float)
        self.diagnostics_dict: Dict[str, Any] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "OLSRegressionModel":
        """Fits OLS model using statsmodels with intercept."""
        self.feature_names = list(X_train.columns)
        X_train_sm = sm.add_constant(X_train, has_constant="add")

        self.fitted_model = sm.OLS(y_train, X_train_sm).fit()
        self.is_fitted = True
        self.summary_table = str(self.fitted_model.summary())

        # Extract coefficients and p-values
        params = self.fitted_model.params
        pvals = self.fitted_model.pvalues
        self.coefficients_ = params[params.index != "const"]
        self.pvalues_ = pvals[pvals.index != "const"]

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts target using fitted OLS model with constant."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_sm = sm.add_constant(X[self.feature_names], has_constant="add")
        # Align columns
        missing_cols = [c for c in self.fitted_model.params.index if c not in X_sm.columns]
        for c in missing_cols:
            X_sm[c] = 0.0
        X_sm = X_sm[self.fitted_model.params.index]
        return self.fitted_model.predict(X_sm).values

    def get_coefficients(self) -> pd.Series:
        """Returns non-intercept feature coefficients."""
        return self.coefficients_

    def get_diagnostics(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Performs Jarque-Bera, Durbin-Watson, Breusch-Pagan, and Condition Number tests."""
        if not self.is_fitted:
            return {}

        residuals = self.fitted_model.resid
        fitted_values = self.fitted_model.fittedvalues

        # 1. Jarque-Bera Normality Test
        jb_stat, jb_pval, skew, kurtosis = jarque_bera(residuals)

        # 2. Durbin-Watson Autocorrelation Test
        dw_stat = float(durbin_watson(residuals))

        # 3. Breusch-Pagan Homoscedasticity Test
        X_train_sm = sm.add_constant(X_train[self.feature_names], has_constant="add")
        try:
            bp_test = het_breuschpagan(residuals, X_train_sm)
            bp_stat, bp_pval = float(bp_test[0]), float(bp_test[1])
        except Exception:
            bp_stat, bp_pval = np.nan, np.nan

        # 4. Condition Number (Multicollinearity)
        condition_number = float(self.fitted_model.condition_number)

        self.diagnostics_dict = {
            "durbin_watson": dw_stat,
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_pval),
            "skewness": float(skew),
            "kurtosis": float(kurtosis),
            "breusch_pagan_stat": bp_stat,
            "breusch_pagan_pvalue": bp_pval,
            "condition_number": condition_number,
            "f_statistic": float(self.fitted_model.fvalue),
            "f_pvalue": float(self.fitted_model.f_pvalue),
            "r2_adj": float(self.fitted_model.rsquared_adj),
            "aic": float(self.fitted_model.aic),
            "bic": float(self.fitted_model.bic)
        }

        return self.diagnostics_dict
