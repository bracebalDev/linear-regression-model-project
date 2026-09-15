"""Stepwise regression with forward-backward feature selection guided by AIC."""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.models.base_model import BaseRegressionModel


class StepwiseAICRegressionModel(BaseRegressionModel):
    """Stepwise regression minimizing Akaike Information Criterion (AIC)."""

    def __init__(
        self,
        model_name: str = "Regresion Stepwise (AIC)",
        max_features: int = 35,
        direction: str = "both",  # 'forward' or 'both'
        verbose: bool = False
    ):
        super().__init__(model_name=model_name)
        self.max_features = max_features
        self.direction = direction
        self.verbose = verbose
        self.selected_features: List[str] = []
        self.aic_history: List[Tuple[int, str, float]] = []
        self.coefficients_: pd.Series = pd.Series(dtype=float)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "StepwiseAICRegressionModel":
        """Executes stepwise forward-backward selection optimizing AIC."""
        self.feature_names = list(X_train.columns)
        remaining_candidates = list(X_train.columns)
        selected: List[str] = []
        
        # Base model with constant only
        n_samples = len(y_train)
        base_sm = sm.OLS(y_train, sm.add_constant(np.ones((n_samples, 1)))).fit()
        current_best_aic = float(base_sm.aic)
        self.aic_history.append((0, "Base (Intercepto)", current_best_aic))

        if self.verbose:
            print(f"[*] Stepwise Base AIC: {current_best_aic:.2f}")

        step = 0
        while remaining_candidates and len(selected) < self.max_features:
            step += 1
            best_candidate = None
            best_candidate_aic = current_best_aic

            # Forward Step: Evaluate adding each candidate
            for candidate in remaining_candidates:
                trial_features = selected + [candidate]
                X_trial = sm.add_constant(X_train[trial_features], has_constant="add")
                try:
                    trial_model = sm.OLS(y_train, X_trial).fit()
                    if trial_model.aic < best_candidate_aic:
                        best_candidate_aic = float(trial_model.aic)
                        best_candidate = candidate
                except Exception:
                    continue

            # Check if forward addition improved AIC
            if best_candidate is not None and best_candidate_aic < current_best_aic:
                selected.append(best_candidate)
                remaining_candidates.remove(best_candidate)
                current_best_aic = best_candidate_aic
                self.aic_history.append((step, best_candidate, current_best_aic))
                if self.verbose:
                    print(f"    [+] Step {step}: Added '{best_candidate}' -> AIC: {current_best_aic:.2f}")
            else:
                # No candidate improves AIC
                break

            # Backward Step (if direction == 'both' and len(selected) > 2)
            if self.direction == "both" and len(selected) > 2:
                worst_feature = None
                backward_best_aic = current_best_aic
                for feat in selected[:-1]:  # Exclude just-added feature
                    sub_features = [f for f in selected if f != feat]
                    X_sub = sm.add_constant(X_train[sub_features], has_constant="add")
                    try:
                        sub_model = sm.OLS(y_train, X_sub).fit()
                        if sub_model.aic < backward_best_aic:
                            backward_best_aic = float(sub_model.aic)
                            worst_feature = feat
                    except Exception:
                        continue

                if worst_feature is not None and backward_best_aic < current_best_aic:
                    selected.remove(worst_feature)
                    remaining_candidates.append(worst_feature)
                    current_best_aic = backward_best_aic
                    self.aic_history.append((step, f"Eliminada '{worst_feature}'", current_best_aic))
                    if self.verbose:
                        print(f"    [-] Step {step}: Removed '{worst_feature}' -> AIC: {current_best_aic:.2f}")

        self.selected_features = selected
        
        # Fit final OLS on selected subset
        X_train_final = sm.add_constant(X_train[self.selected_features], has_constant="add")
        self.fitted_model = sm.OLS(y_train, X_train_final).fit()
        self.is_fitted = True

        params = self.fitted_model.params
        self.coefficients_ = params[params.index != "const"]

        print(f"[+] Stepwise AIC converged with {len(self.selected_features)} features. Final AIC: {current_best_aic:.2f}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts target using the fitted Stepwise subset model."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")
        X_sub = sm.add_constant(X[self.selected_features], has_constant="add")
        return self.fitted_model.predict(X_sub).values

    def get_coefficients(self) -> pd.Series:
        """Returns coefficients for selected features."""
        return self.coefficients_

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Returns Stepwise selection details."""
        return {
            "criterio": "AIC",
            "num_variables_seleccionadas": len(self.selected_features),
            "variables_seleccionadas": self.selected_features,
            "aic_final": self.fitted_model.aic if self.fitted_model else None
        }
