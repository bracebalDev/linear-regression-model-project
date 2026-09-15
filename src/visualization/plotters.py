"""Publication-quality visualization routines for industrial data analytics."""

import matplotlib
matplotlib.use("Agg")
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.config import CONFIG, ProjectConfig


class IndustrialVisualizer:
    """Generates charts for EDA, econometric diagnostics, and model performance."""

    def __init__(self, config: ProjectConfig = CONFIG):
        self.config = config
        self.figures_dir = config.FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Apply standard clean styling
        sns.set_theme(style="whitegrid", font="sans-serif")
        plt.rcParams.update({
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.autolayout": True
        })

    def plot_target_distribution(self, y: pd.Series, target_name: str = "CONSUMO ESPECIFICO CLO2") -> Path:
        """Plots distribution (KDE + Histogram) and Boxplot of the target variable with 17.5 kg/ADt design line."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram + KDE
        sns.histplot(y, kde=True, ax=axes[0], color="#1f77b4", bins=40, edgecolor="black", alpha=0.6)
        axes[0].axvline(self.config.DESIGN_CONSUMPTION_LIMIT, color="red", linestyle="--", linewidth=2, 
                        label=f"Límite Diseño ({self.config.DESIGN_CONSUMPTION_LIMIT} kg/ADt)")
        axes[0].axvline(y.mean(), color="green", linestyle="-", linewidth=2, 
                        label=f"Media Observada ({y.mean():.2f} kg/ADt)")
        axes[0].set_title(f"Distribución de {target_name}", fontweight="bold")
        axes[0].set_xlabel("Consumo Específico $ClO_2$ (kg/ADt)")
        axes[0].set_ylabel("Frecuencia")
        axes[0].legend()

        # Boxplot
        sns.boxplot(x=y, ax=axes[1], color="#90caf9", showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"})
        axes[1].axvline(self.config.DESIGN_CONSUMPTION_LIMIT, color="red", linestyle="--", linewidth=2, 
                        label="Límite Diseño (17.5 kg/ADt)")
        axes[1].set_title(f"Boxplot de Variabilidad y Outliers - {target_name}", fontweight="bold")
        axes[1].set_xlabel("Consumo Específico $ClO_2$ (kg/ADt)")
        axes[1].legend()

        output_file = self.figures_dir / "01_eda_target_distribution.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"[+] Saved target distribution figure to: {output_file}")
        return output_file

    def plot_timeseries_overview(self, df: pd.DataFrame, target_col: str = "CONSUMO ESPECIFICO CLO2") -> Path:
        """Plots the full high-frequency (2-minute) industrial time series with operational limits."""
        fig, ax = plt.subplots(figsize=(15, 6))

        if isinstance(df.index, pd.DatetimeIndex):
            ax.plot(df.index, df[target_col], label="Consumo Medido (2 min)", color="#2b5c8f", alpha=0.7, linewidth=0.8)
            # Rolling 24-hour mean
            rolling_daily = df[target_col].rolling(window=720, min_periods=60).mean()
            ax.plot(df.index, rolling_daily, label="Media Móvil 24h", color="#e65100", linewidth=2.2)
            ax.set_xlabel("Fecha (Abril 2019)")
        else:
            ax.plot(df[target_col].values, label="Consumo Medido", color="#2b5c8f", alpha=0.7, linewidth=0.8)
            ax.set_xlabel("Índice de Muestra")

        ax.axhline(self.config.DESIGN_CONSUMPTION_LIMIT, color="red", linestyle="--", linewidth=2,
                   label=f"Límite de Diseño ({self.config.DESIGN_CONSUMPTION_LIMIT} kg/ADt)")
        ax.set_ylabel("Consumo $ClO_2$ (kg/ADt)")
        ax.set_title("Serie Temporal del Proceso de Blanqueo - Consumo Específico de Dióxido de Cloro", fontweight="bold")
        ax.legend(loc="upper right")

        output_file = self.figures_dir / "02_eda_timeseries_trend.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"[+] Saved timeseries trend figure to: {output_file}")
        return output_file

    def plot_correlation_heatmap(self, df: pd.DataFrame, target_col: str, top_n: int = 18) -> Path:
        """Plots correlation matrix of top features correlated with target."""
        numeric_df = df.select_dtypes(include=np.number)
        corr_series = numeric_df.corr()[target_col].abs().sort_values(ascending=False)
        top_cols = corr_series.head(top_n).index.tolist()

        sub_corr = numeric_df[top_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(sub_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax, linewidths=0.5,
                    square=True, annot_kws={"size": 8})
        ax.set_title(f"Matriz de Correlación de las Top {top_n} Variables más Influyentes", fontweight="bold")

        output_file = self.figures_dir / "03_correlation_heatmap.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"[+] Saved correlation heatmap to: {output_file}")
        return output_file

    def plot_ols_diagnostics(
        self,
        y_true: pd.Series,
        y_fitted: np.ndarray,
        residuals: np.ndarray
    ) -> Path:
        """Plots 4 standard econometric diagnostic panels for OLS assumptions."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # Panel 1: Residuals vs Fitted (Linearity & Homoscedasticity)
        axes[0, 0].scatter(y_fitted, residuals, alpha=0.3, color="#1976d2", s=15)
        axes[0, 0].axhline(0, color="red", linestyle="--", linewidth=1.5)
        sns.regplot(x=y_fitted, y=residuals, scatter=False, lowess=True, ax=axes[0, 0], color="orange")
        axes[0, 0].set_title("1. Residuos vs Valores Ajustados (Linealidad)", fontweight="bold")
        axes[0, 0].set_xlabel("Valores Ajustados ($\\hat{y}$)")
        axes[0, 0].set_ylabel("Residuos ($e_i$)")

        # Panel 2: Normal Q-Q Plot (Residual Normality)
        sm.qqplot(residuals, line="45", fit=True, ax=axes[0, 1], alpha=0.3, color="#388e3c")
        axes[0, 1].set_title("2. Q-Q Plot de Residuos (Normalidad)", fontweight="bold")
        axes[0, 1].set_xlabel("Cuantiles Teóricos Normales")
        axes[0, 1].set_ylabel("Cuantiles Muestrales de Residuos")

        # Panel 3: Scale-Location (Homoscedasticity)
        standardized_resids = (residuals - np.mean(residuals)) / np.std(residuals)
        sqrt_abs_resids = np.sqrt(np.abs(standardized_resids))
        axes[1, 0].scatter(y_fitted, sqrt_abs_resids, alpha=0.3, color="#7b1fa2", s=15)
        sns.regplot(x=y_fitted, y=sqrt_abs_resids, scatter=False, lowess=True, ax=axes[1, 0], color="red")
        axes[1, 0].set_title("3. Scale-Location (Homocedasticidad)", fontweight="bold")
        axes[1, 0].set_xlabel("Valores Ajustados ($\\hat{y}$)")
        axes[1, 0].set_ylabel("$\\sqrt{|Residuos\\ Estándar|}$")

        # Panel 4: Residuals vs Time / Sequence (Autocorrelation)
        axes[1, 1].plot(residuals[:1000], color="#c2185b", linewidth=0.8, alpha=0.8)
        axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
        axes[1, 1].set_title("4. Secuencia de Residuos (Autocorrelación - Muestra 1000)", fontweight="bold")
        axes[1, 1].set_xlabel("Paso Temporal / Muestra")
        axes[1, 1].set_ylabel("Residuo ($e_t$)")

        output_file = self.figures_dir / "04_ols_residuals_vs_fitted.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"[+] Saved OLS econometric diagnostic panels to: {output_file}")
        return output_file

    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> Path:
        """Plots comparative bar charts of Test R2, RMSE, and retained features across all models."""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        models = comparison_df["Modelo"]

        # Panel 1: Test R2
        sns.barplot(x="Test R2", y="Modelo", data=comparison_df, ax=axes[0], palette="Blues_r")
        axes[0].set_title("Coeficiente de Determinación ($R^2$ Test)", fontweight="bold")
        axes[0].set_xlim(min(0.85, comparison_df["Test R2"].min() - 0.05), 1.0)
        for i, v in enumerate(comparison_df["Test R2"]):
            axes[0].text(v - 0.04, i, f"{v:.4f}", color="white", fontweight="bold", va="center")

        # Panel 2: Test RMSE
        sns.barplot(x="Test RMSE (kg/ADt)", y="Modelo", data=comparison_df, ax=axes[1], palette="Reds_r")
        axes[1].set_title("Error Cuadrático Medio (RMSE Test)", fontweight="bold")
        for i, v in enumerate(comparison_df["Test RMSE (kg/ADt)"]):
            axes[1].text(v * 0.5, i, f"{v:.4f}", color="black", fontweight="bold", va="center")

        # Panel 3: Number of Features
        sns.barplot(x="N Variables", y="Modelo", data=comparison_df, ax=axes[2], palette="Greens_r")
        axes[2].set_title("Parsimonia (N° Variables Retenidas)", fontweight="bold")
        for i, v in enumerate(comparison_df["N Variables"]):
            axes[2].text(v * 0.5, i, f"{int(v)}", color="black", fontweight="bold", va="center")

        output_file = self.figures_dir / "06_model_comparison_metrics.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"[+] Saved model comparison bar charts to: {output_file}")
        return output_file

    def plot_feature_importance(self, coefficients: pd.Series, model_name: str, top_n: int = 15) -> Path:
        """Plots horizontal bar chart of the highest magnitude regression coefficients."""
        clean_coefs = coefficients[coefficients != 0].copy()
        top_pos = clean_coefs.sort_values(ascending=False).head(top_n // 2)
        top_neg = clean_coefs.sort_values(ascending=True).head(top_n // 2)
        top_subset = pd.concat([top_pos, top_neg]).sort_values()

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#d32f2f" if val < 0 else "#1976d2" for val in top_subset.values]
        ax.barh(top_subset.index, top_subset.values, color=colors, edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Variables más Influyentes en Consumo $ClO_2$ - {model_name}", fontweight="bold")
        ax.set_xlabel("Magnitud del Coeficiente Estandarizado ($\\beta_j$)")

        output_file = self.figures_dir / "07_feature_importance_lasso_ridge_elasticnet.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"[+] Saved feature importance plot to: {output_file}")
        return output_file

    def plot_stepwise_aic_curve(self, aic_history: List[Tuple[int, str, float]]) -> Path:
        """Plots the AIC minimization curve across Stepwise selection steps."""
        steps = [h[0] for h in aic_history]
        aics = [h[2] for h in aic_history]
        labels = [h[1] for h in aic_history]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps, aics, marker="o", color="#d81b60", linewidth=2.2, markersize=5)
        ax.set_title("Trayectoria de Optimización AIC en Regresión Stepwise", fontweight="bold")
        ax.set_xlabel("Paso de Selección (Iteración)")
        ax.set_ylabel("Akaike Information Criterion (AIC)")
        ax.grid(True)

        output_file = self.figures_dir / "08_stepwise_aic_trajectory.png"
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"[+] Saved Stepwise AIC trajectory to: {output_file}")
        return output_file
