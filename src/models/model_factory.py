"""Model Factory for dynamic regression model instantiation (Factory Pattern)."""

from typing import Dict, Type
from src.models.base_model import BaseRegressionModel
from src.models.ols_regression import OLSRegressionModel
from src.models.stepwise_regression import StepwiseAICRegressionModel
from src.models.regularized_models import (
    RidgeRegressionModel,
    LassoRegressionModel,
    ElasticNetRegressionModel,
)


class ModelFactory:
    """Factory creating regression models with uniform lifecycle interface."""

    _REGISTRY: Dict[str, Type[BaseRegressionModel]] = {
        "ols": OLSRegressionModel,
        "stepwise": StepwiseAICRegressionModel,
        "ridge": RidgeRegressionModel,
        "lasso": LassoRegressionModel,
        "elasticnet": ElasticNetRegressionModel,
    }

    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseRegressionModel:
        """Instantiates a model strategy by registered key."""
        key = model_type.lower().strip()
        if key not in cls._REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. Available types: {list(cls._REGISTRY.keys())}"
            )
        model_cls = cls._REGISTRY[key]
        return model_cls(**kwargs)

    @classmethod
    def get_all_models(cls) -> Dict[str, BaseRegressionModel]:
        """Instantiates default suite of all 5 regression models."""
        return {
            "OLS": OLSRegressionModel(),
            "Stepwise": StepwiseAICRegressionModel(),
            "Ridge": RidgeRegressionModel(),
            "Lasso": LassoRegressionModel(),
            "ElasticNet": ElasticNetRegressionModel(),
        }
