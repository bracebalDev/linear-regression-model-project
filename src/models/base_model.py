"""Abstract base class and contract for all regression models (Strategy Pattern)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


@dataclass
class ModelEvaluationMetrics:
    """Standardized metrics container across all linear regression variants."""
    model_name: str
    train_mse: float
    train_rmse: float
    train_mae: float
    train_r2: float
    test_mse: float
    test_rmse: float
    test_mae: float
    test_r2: float
    adjusted_r2: float
    aic: Optional[float]
    bic: Optional[float]
    num_features_retained: int
    optimal_hyperparameters: Dict[str, Any]
    additional_diagnostics: Dict[str, Any]


class BaseRegressionModel(ABC):
    """Abstract Strategy interface for all predictive and explanatory regression models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.fitted_model: Any = None

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseRegressionModel":
        """Trains the regression model on given features and target."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions for given input features."""
        pass

    @abstractmethod
    def get_coefficients(self) -> pd.Series:
        """Returns the learned model coefficients mapped to feature names."""
        pass

    def evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> ModelEvaluationMetrics:
        """Standardized evaluation computing Train/Test errors, R2, Adjusted R2, AIC, and BIC."""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.model_name}' must be fitted before evaluation.")

        # Train predictions and errors
        y_train_pred = self.predict(X_train)
        train_mse = float(mean_squared_error(y_train, y_train_pred))
        train_rmse = float(np.sqrt(train_mse))
        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        train_r2 = float(r2_score(y_train, y_train_pred))

        # Test predictions and errors
        y_test_pred = self.predict(X_test)
        test_mse = float(mean_squared_error(y_test, y_test_pred))
        test_rmse = float(np.sqrt(test_mse))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))

        # Degrees of freedom for Adjusted R2
        n_test = len(y_test)
        coefs = self.get_coefficients()
        k_features = int((coefs != 0).sum()) if coefs is not None else X_train.shape[1]
        
        if n_test > k_features + 1:
            adjusted_r2 = 1.0 - (1.0 - test_r2) * (n_test - 1) / (n_test - k_features - 1)
        else:
            adjusted_r2 = test_r2

        # Information Criteria (AIC / BIC)
        n_train = len(y_train)
        rss_train = np.sum((y_train.values.ravel() - y_train_pred.ravel()) ** 2)
        if rss_train > 0:
            k_params = k_features + 1  # Including intercept
            aic = float(n_train * np.log(rss_train / n_train) + 2 * k_params)
            bic = float(n_train * np.log(rss_train / n_train) + np.log(n_train) * k_params)
        else:
            aic, bic = None, None

        diagnostics = self.get_diagnostics(X_train, y_train)

        return ModelEvaluationMetrics(
            model_name=self.model_name,
            train_mse=train_mse,
            train_rmse=train_rmse,
            train_mae=train_mae,
            train_r2=train_r2,
            test_mse=test_mse,
            test_rmse=test_rmse,
            test_mae=test_mae,
            test_r2=test_r2,
            adjusted_r2=float(adjusted_r2),
            aic=aic,
            bic=bic,
            num_features_retained=k_features,
            optimal_hyperparameters=self.get_hyperparameters(),
            additional_diagnostics=diagnostics
        )

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Returns dictionary of optimal tuned parameters."""
        return {}

    def get_diagnostics(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Hook method for custom diagnostic tests (normality, heteroscedasticity, etc.)."""
        return {}
