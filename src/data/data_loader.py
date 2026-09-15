"""Industrial Data Loader module for multi-tier Excel and Parquet datasets."""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional
import pandas as pd
from src.config import CONFIG, ProjectConfig


class IndustrialDataLoader:
    """Robust loader for industrial plant time-series datasets with multi-tier headers."""

    def __init__(self, config: ProjectConfig = CONFIG):
        self.config = config
        self.tag_metadata: Dict[str, str] = {}
        self.unit_metadata: Dict[str, str] = {}

    def get_raw_file_path(self) -> Path:
        """Resolves the raw data path checking primary and fallback locations."""
        if self.config.RAW_DATA_PATH.exists():
            return self.config.RAW_DATA_PATH
        if self.config.FALLBACK_RAW_DATA_PATH.exists():
            return self.config.FALLBACK_RAW_DATA_PATH
        raise FileNotFoundError(
            f"Raw dataset not found at '{self.config.RAW_DATA_PATH}' nor '{self.config.FALLBACK_RAW_DATA_PATH}'."
        )

    def load_raw_data(self, use_cache: bool = True) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
        """
        Loads raw Excel data, parses sensor tags and engineering units, 
        and extracts 2-minute interval timestamp records.
        """
        raw_path = self.get_raw_file_path()
        cached_parquet = self.config.CACHED_PARQUET_PATH

        # If cache exists and use_cache is True, read cache for ultra fast I/O
        if use_cache and cached_parquet.exists():
            df_cached = pd.read_parquet(cached_parquet)
            return df_cached, self.tag_metadata, self.unit_metadata

        print(f"[*] Loading raw industrial Excel dataset from: {raw_path}")
        df_raw = pd.read_excel(raw_path)

        # Row 0 contains plant sensor DCS tag codes (e.g. CE05_IP21_547AI1133)
        # Row 1 contains engineering units (e.g. kg/s, kg/ADt, %, uS/cm)
        for col_name in df_raw.columns:
            tag_code = str(df_raw.iloc[0][col_name]) if pd.notnull(df_raw.iloc[0][col_name]) else "N/A"
            unit_val = str(df_raw.iloc[1][col_name]) if pd.notnull(df_raw.iloc[1][col_name]) else "N/A"
            self.tag_metadata[str(col_name)] = tag_code
            self.unit_metadata[str(col_name)] = unit_val

        # Actual observational data begins at index 2
        df_data = df_raw.iloc[2:].copy()

        # Drop empty unnamed columns if present
        cols_to_drop = [c for c in ["Unnamed: 0", "Unnamed: 1"] if c in df_data.columns]
        if cols_to_drop:
            df_data.drop(columns=cols_to_drop, inplace=True)

        # Parse timestamp
        if "Unnamed: 2" in df_data.columns:
            df_data.rename(columns={"Unnamed: 2": "Timestamp"}, inplace=True)
            df_data["Timestamp"] = pd.to_datetime(df_data["Timestamp"])
            df_data.set_index("Timestamp", inplace=True)

        # Cast all sensor columns to numeric
        for col in df_data.columns:
            df_data[col] = pd.to_numeric(df_data[col], errors="coerce")

        # Sort index chronologically
        df_data.sort_index(inplace=True)

        # Cache clean dataset
        os.makedirs(self.config.DATA_PROCESSED_DIR, exist_ok=True)
        try:
            df_data.to_parquet(cached_parquet)
            df_data.to_csv(self.config.CLEAN_DATASET_CSV)
            print(f"[+] Cached processed dataset to '{cached_parquet}' and '{self.config.CLEAN_DATASET_CSV}'")
        except Exception as e:
            print(f"[!] Warning: Could not cache dataset: {e}")

        print(f"[+] Loaded {df_data.shape[0]} records and {df_data.shape[1]} sensor columns.")
        return df_data, self.tag_metadata, self.unit_metadata
