"""Configuration and global constants for the Cellulose Bleaching Analytics Project."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ProjectConfig:
    """Project-wide configuration container with mnemonic constants."""

    # Base Paths
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = ROOT_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = ROOT_DIR / "data" / "processed"
    REPORTS_DIR: Path = ROOT_DIR / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"
    
    # File Paths
    RAW_EXCEL_FILENAME: str = "Ab19selec.xlsx"
    RAW_DATA_PATH: Path = DATA_RAW_DIR / RAW_EXCEL_FILENAME
    FALLBACK_RAW_DATA_PATH: Path = ROOT_DIR / RAW_EXCEL_FILENAME
    CACHED_PARQUET_PATH: Path = DATA_PROCESSED_DIR / "industrial_bleaching_data.parquet"
    CLEAN_DATASET_CSV: Path = DATA_PROCESSED_DIR / "industrial_bleaching_clean.csv"

    # Reproducibility & Statistical Settings
    RANDOM_SEED: int = 2022
    TEST_SPLIT_RATIO: float = 0.20
    CV_FOLDS: int = 5
    
    # Industrial Process Targets & Benchmarks
    PRIMARY_TARGET_VARIABLE: str = "CONSUMO ESPECIFICO CLO2"
    SECONDARY_TARGET_VARIABLE: str = "Consumo Total ClO2"
    DESIGN_CONSUMPTION_LIMIT: float = 17.5  # kg/ADt (Design operational limit)
    HISTORICAL_AVG_CONSUMPTION: float = 19.10  # kg/ADt
    HISTORICAL_MAX_CONSUMPTION: float = 20.37  # kg/ADt
    POTENTIAL_ANNUAL_SAVINGS_USD: float = 1_600_000.0
    OPTIMIZATION_BENEFIT_30PCT_USD: float = 500_000.0

    # Data Leakage / Direct Arithmetic Component Columns to exclude from predictive features
    DATA_LEAKAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        "Consumo Total ClO2",
        "Consumo ClO2 Etapa D0 L1",
        "Consumo ClO2 Etapa D1 L1",
        "Consumo ClO2 Etapa D2 L1",
        "Consumo ClO2 L1",
        "Consumo ClO2 Etapa D0",
        "Consumo ClO2 Etapa D1",
        "Consumo ClO2 Etapa D2",
        "CONSUMO ESPECIFICO CLO2",
        "CUMPLE O  NO CUMPLE"
    ])

    # Columns with zero variance or known constant sensors
    CONSTANT_OR_DISABLED_COLUMNS: List[str] = field(default_factory=lambda: [
        "Unnamed: 0",
        "Unnamed: 1",
        "CE05_IP21_547FI1019",
        "Flujo H2SO4 Etapa D0",
        "Agua de Lavado prensa preblanqueo",
        "CUMPLE O  NO CUMPLE"
    ])

    # Process Stages in Chile Pulp Bleaching
    PROCESS_STAGES: List[str] = field(default_factory=lambda: [
        "Pre-Blanqueo",
        "Etapa D0",
        "Etapa EOP",
        "Etapa D1",
        "Etapa D2 y Almacenamiento"
    ])


# Default Singleton Instance
CONFIG = ProjectConfig()
