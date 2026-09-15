"""Regularized Linear Models: Ridge (L2), Lasso (L1), and Elastic Net (L1+L2)."""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from src.models.base_model import BaseRegressionModel
from src.config import CONFIG


class RidgeRegressionModel(BaseRegressionModel):
    """Ridge Regression (L2 Regularization) with Cross-Validated optimal lambda search."""

    def __init__(
        self,
        model_name: str = "Regresion Ridge (L2)",
        alphas: Optional[np.ndarray] = None,
        cv: int = CONFIG.CV_FOLDS
    ):
        super().__init__(model_name=model_name)
        self.alphas = alphas if alphas is not None else np.logspace(-4, 4, 60)
        self.cv = cv
        self.optimal_lambda_: float = 0.0
        self.coefficients_: pd.Series = pd.Series(dtype=float)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "RidgeRegressionModel":
        """Fits RidgeCV with 5-fold cross validation."""
        self.feature_names = list(X_train.columns)
        self.fitted_model = RidgeCV(
            alphas=self.alphas,
            cv=self.cv,
            scoring="neg_mean_squared_error"
        )
        self.fitted_model.fit(X_train, y_train)
        self.optimal_lambda_ = float(self.fitted_model.alpha_)
        self.coefficients_ = pd.Series(self.fitted_model.coef_, index=self.feature_names)
        self.is_fitted = True

        print(f"[+] Ridge Regression fitted. Optimal lambda (alpha) = {self.optimal_lambda_:.6f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts target using fitted Ridge model."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.fitted_model.predict(X[self.feature_names])

    def get_coefficients(self) -> pd.Series:
        """Returns Ridge regularized feature coefficients."""
        return self.coefficients_

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Returns optimal lambda parameter."""
        return {
            "regularizacion": "L2 (Ridge)",
            "lambda_optimo": self.optimal_lambda_,
            "cv_folds": self.cv
        }


class LassoRegressionModel(BaseRegressionModel):
    """Lasso Regression (L1 Regularization) with Cross-Validated optimal lambda and sparsity selection."""

    def __init__(
        self,
        model_name: str = "Regresion Lasso (L1)",
        cv: int = CONFIG.CV_FOLDS,
        random_state: int = CONFIG.RANDOM_SEED,
        max_iter: int = 15000
    ):
        super().__init__(model_name=model_name)
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.optimal_lambda_: float = 0.0
        self.coefficients_: pd.Series = pd.Series(dtype=float)
        self.selected_features_: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LassoRegressionModel":
        """Fits LassoCV finding optimal lambda with 5-fold CV."""
        self.feature_names = list(X_train.columns)
        self.fitted_model = LassoCV(
            cv=self.cv,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_jobs=-1
        )
        self.fitted_model.fit(X_train, y_train)
        self.optimal_lambda_ = float(self.fitted_model.alpha_)
        self.coefficients_ = pd.Series(self.fitted_model.coef_, index=self.feature_names)
        self.selected_features_ = list(self.coefficients_[self.coefficients_ != 0].index)
        self.is_fitted = True

        print(
            f"[+] Lasso Regression fitted. Optimal lambda = {self.optimal_lambda_:.6f}, "
            f"Features selected: {len(self.selected_features_)}/{len(self.feature_names)}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts target using fitted Lasso model."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.fitted_model.predict(X[self.feature_names])

    def get_coefficients(self) -> pd.Series:
        """Returns Lasso coefficients (with exact zeros for discarded features)."""
        return self.coefficients_

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Returns optimal lambda and sparsity statistics."""
        return {
            "regularizacion": "L1 (Lasso)",
            "lambda_optimo": self.optimal_lambda_,
            "num_variables_seleccionadas": len(self.selected_features_),
            "sparsity_pct": float(100 * (1 - len(self.selected_features_) / len(self.feature_names))),
            "cv_folds": self.cv
        }


class ElasticNetRegressionModel(BaseRegressionModel):
    """Elastic Net Regression (L1 + L2 Regularization) with optimal lambda and mixing ratio alpha."""

    def __init__(
        self,
        model_name: str = "Regresion Elastic Net (L1+L2)",
        l1_ratios: Optional[List[float]] = None,
        cv: int = CONFIG.CV_FOLDS,
        random_state: int = CONFIG.RANDOM_SEED,
        max_iter: int = 15000
    ):
        super().__init__(model_name=model_name)
        self.l1_ratios = l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.optimal_lambda_: float = 0.0
        self.optimal_l1_ratio_: float = 0.0
        self.coefficients_: pd.Series = pd.Series(dtype=float)
        self.selected_features_: List[str] = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "ElasticNetRegressionModel":
        """Fits ElasticNetCV finding both optimal lambda and optimal L1 mixing ratio."""
        self.feature_names = list(X_train.columns)
        self.fitted_model = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=self.cv,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_jobs=-1
        )
        self.fitted_model.fit(X_train, y_train)
        self.optimal_lambda_ = float(self.fitted_model.alpha_)
        self.optimal_l1_ratio_ = float(self.fitted_model.l1_ratio_)
        self.coefficients_ = pd.Series(self.fitted_model.coef_, index=self.feature_names)
        self.selected_features_ = list(self.coefficients_[self.coefficients_ != 0].index)
        self.is_fitted = True

        print(
            f"[+] ElasticNet fitted. Optimal lambda = {self.optimal_lambda_:.6f}, "
            f"Optimal L1 ratio (alpha mezcla) = {self.optimal_l1_ratio_:.2f}, "
            f"Features selected: {len(self.selected_features_)}/{len(self.feature_names)}"
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts target using fitted ElasticNet model."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        return self.fitted_model.predict(X[self.feature_names])

    def get_coefficients(self) -> pd.Series:
        """Returns Elastic Net coefficients."""
        return self.coefficients_

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Returns optimal lambda and mixing ratio."""
        return {
            "regularizacion": "Elastic Net (L1+L2)",
            "lambda_optimo": self.optimal_lambda_,
            "l1_ratio_optimo": self.optimal_l1_ratio_,
            "num_variables_seleccionadas": len(self.selected_features_),
            "sparsity_pct": float(100 * (1 - len(self.selected_features_) / len(self.feature_names))),
            "cv_folds": self.cv
        }
