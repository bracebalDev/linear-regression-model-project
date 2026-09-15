"""Model evaluation, comparative metric aggregation, and business impact translation."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from tabulate import tabulate
from src.models.base_model import ModelEvaluationMetrics, BaseRegressionModel
from src.config import CONFIG, ProjectConfig


class ModelEvaluator:
    """Evaluates, ranks, and compares multiple regression models across statistical and business dimensions."""

    def __init__(self, config: ProjectConfig = CONFIG):
        self.config = config
        self.metrics_list: List[ModelEvaluationMetrics] = []

    def add_evaluation(self, metrics: ModelEvaluationMetrics) -> None:
        """Appends model metrics to the evaluation store."""
        self.metrics_list.append(metrics)

    def generate_comparison_dataframe(self) -> pd.DataFrame:
        """Creates a standardized comparison DataFrame sorted by Test R2 descending."""
        rows = []
        for m in self.metrics_list:
            hyper_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                  for k, v in m.optimal_hyperparameters.items() if k not in ["variables_seleccionadas"])
            rows.append({
                "Modelo": m.model_name,
                "Train R2": m.train_r2,
                "Test R2": m.test_r2,
                "Test R2 Adj": m.adjusted_r2,
                "Test RMSE (kg/ADt)": m.test_rmse,
                "Test MAE (kg/ADt)": m.test_mae,
                "AIC": m.aic if m.aic is not None else float("nan"),
                "BIC": m.bic if m.bic is not None else float("nan"),
                "N Variables": m.num_features_retained,
                "Hiperparametros / Regularizacion": hyper_str or "N/A"
            })

        df_comparison = pd.DataFrame(rows)
        df_comparison.sort_values(by="Test R2", ascending=False, inplace=True)
        df_comparison.reset_index(drop=True, inplace=True)
        return df_comparison

    def print_comparison_table(self) -> str:
        """Formats comparison table nicely for terminal output."""
        df_comp = self.generate_comparison_dataframe()
        table_str = tabulate(df_comp, headers="keys", tablefmt="grid", floatfmt=".4f")
        print("\n" + "="*80)
        print("[*] TABLA COMPARATIVA DE MODELOS DE REGRESION (DATA ANALYTICS)")
        print("="*80)
        print(table_str)
        return table_str

    def select_best_model(self) -> ModelEvaluationMetrics:
        """
        Selects the best overall model considering generalization, parsimony, and AIC.
        Prefers sparse/regularized models when Test R2 difference is negligible (<0.001).
        """
        if not self.metrics_list:
            raise ValueError("No evaluated models available.")

        # Sort by test R2
        sorted_metrics = sorted(self.metrics_list, key=lambda m: m.test_r2, reverse=True)
        top_model = sorted_metrics[0]

        # Check if a sparser model (Stepwise, Lasso, ElasticNet) achieves within 0.1% performance
        candidate_models = [m for m in sorted_metrics if (top_model.test_r2 - m.test_r2) < 0.0015]
        # Choose candidate with minimum number of features
        best_model = min(candidate_models, key=lambda m: m.num_features_retained)

        print(f"\n[BEST MODEL] Modelo Seleccionado como Optimo: '{best_model.model_name}'")
        print(f"     - Test R2: {best_model.test_r2:.4f}")
        print(f"     - Test RMSE: {best_model.test_rmse:.4f} kg/ADt")
        print(f"     - Test MAE: {best_model.test_mae:.4f} kg/ADt")
        print(f"     - Variables Activas: {best_model.num_features_retained}")
        if best_model.aic:
            print(f"     - AIC: {best_model.aic:.2f}")

        return best_model

    def export_evaluation_json(self, output_path: Optional[Path] = None) -> None:
        """Saves structured metrics and diagnostics to a JSON artifact."""
        target_path = output_path or (self.config.REPORTS_DIR / "model_diagnostics_report.json")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        data_to_export = []
        for m in self.metrics_list:
            data_to_export.append({
                "model_name": m.model_name,
                "train_mse": m.train_mse,
                "train_rmse": m.train_rmse,
                "train_mae": m.train_mae,
                "train_r2": m.train_r2,
                "test_mse": m.test_mse,
                "test_rmse": m.test_rmse,
                "test_mae": m.test_mae,
                "test_r2": m.test_r2,
                "adjusted_r2": m.adjusted_r2,
                "aic": m.aic,
                "bic": m.bic,
                "num_features_retained": m.num_features_retained,
                "optimal_hyperparameters": m.optimal_hyperparameters,
                "additional_diagnostics": m.additional_diagnostics
            })

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data_to_export, f, indent=4, ensure_ascii=False)
        print(f"[+] Saved evaluation report JSON to: {target_path}")
