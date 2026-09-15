"""Data Preprocessor module for cleaning, leakage prevention, split, and scaling."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import CONFIG, ProjectConfig


@dataclass
class PreprocessedData:
    """Container for preprocessed train and test data splits with feature metadata."""
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    feature_names: List[str]
    target_name: str
    dropped_constant_columns: List[str]
    dropped_leakage_columns: List[str]


class IndustrialDataPreprocessor:
    """Preprocessor handling zero-variance removal, leakage filtering, median imputation, and scaling."""

    def __init__(self, config: ProjectConfig = CONFIG):
        self.config = config

    def clean_and_prepare(
        self,
        dataframe: pd.DataFrame,
        target_variable: Optional[str] = None
    ) -> PreprocessedData:
        """
        Executes full preprocessing pipeline:
        1. Validates target variable presence.
        2. Detects and removes zero-variance or constant columns.
        3. Filters out direct mathematical leakage features.
        4. Imputes missing sensor measurements using train median.
        5. Splits into 80% Train / 20% Test (random_state=2022).
        6. Standardizes features with StandardScaler without leakage.
        """
        target_col = target_variable or self.config.PRIMARY_TARGET_VARIABLE
        if target_col not in dataframe.columns:
            # Fallback to secondary target if primary not found
            if self.config.SECONDARY_TARGET_VARIABLE in dataframe.columns:
                print(f"[!] Target '{target_col}' not found. Falling back to '{self.config.SECONDARY_TARGET_VARIABLE}'.")
                target_col = self.config.SECONDARY_TARGET_VARIABLE
            else:
                raise ValueError(f"Target variable '{target_col}' not present in dataframe columns.")

        df_work = dataframe.copy()

        # Step 1: Detect constant or zero-variance columns
        constant_cols = []
        for col in df_work.columns:
            if col == target_col:
                continue
            if df_work[col].nunique(dropna=True) <= 1 or df_work[col].std(ddof=1) <= 1e-7:
                constant_cols.append(col)

        # Include explicitly known disabled columns
        for col in self.config.CONSTANT_OR_DISABLED_COLUMNS:
            if col in df_work.columns and col not in constant_cols and col != target_col:
                constant_cols.append(col)

        constant_cols = list(set(constant_cols))
        print(f"[*] Detected {len(constant_cols)} constant or uninformative columns to drop: {constant_cols}")

        # Step 2: Identify and isolate leakage columns
        leakage_cols = [c for c in self.config.DATA_LEAKAGE_COLUMNS if c in df_work.columns and c != target_col]
        print(f"[*] Identified {len(leakage_cols)} arithmetic leakage columns to isolate from predictive features: {leakage_cols}")

        # Define candidate feature columns
        excluded_cols = set(constant_cols + leakage_cols + [target_col])
        feature_cols = [c for c in df_work.columns if c not in excluded_cols]

        # Extract X and y
        X = df_work[feature_cols].copy()
        y = df_work[target_col].copy()

        # Step 3: Handle target NaNs if any
        if y.isnull().any():
            valid_mask = y.notnull()
            X = X[valid_mask]
            y = y[valid_mask]

        # Step 4: Train / Test Split (80% / 20%, Seed=2022)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SPLIT_RATIO,
            random_state=self.config.RANDOM_SEED,
            shuffle=True
        )

        # Step 5: Median Imputation based strictly on X_train medians
        train_medians = X_train.median()
        X_train_imputed = X_train.fillna(train_medians)
        X_test_imputed = X_test.fillna(train_medians)

        # Step 6: Feature Standardization (StandardScaler fitted on X_train only)
        scaler = StandardScaler()
        X_train_scaled_array = scaler.fit_transform(X_train_imputed)
        X_test_scaled_array = scaler.transform(X_test_imputed)

        X_train_scaled = pd.DataFrame(
            X_train_scaled_array,
            columns=feature_cols,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled_array,
            columns=feature_cols,
            index=X_test.index
        )

        print(f"[+] Dataset preprocessed successfully: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")

        return PreprocessedData(
            X_train_raw=X_train_imputed,
            X_test_raw=X_test_imputed,
            X_train_scaled=X_train_scaled,
            X_test_scaled=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            scaler=scaler,
            feature_names=feature_cols,
            target_name=target_col,
            dropped_constant_columns=constant_cols,
            dropped_leakage_columns=leakage_cols
        )
