"""Domain feature classification and multicollinearity diagnostics."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


class DomainFeatureClassifier:
    """Classifies industrial variables into the 5 domain categories specified in Proyecto.pdf."""

    CATEGORY_MAPPING = {
        "sin_medicion": "Variables sin medición (inhabilitadas o nulas)",
        "control_elementos": "Variables Control de Elementos (motores, válvulas, actuadores)",
        "valor_fijo": "Variables de valor fijo (constantes / varianza cero)",
        "criterio_experto": "Variables Criterio Experto (calidad de pulpa: Kappa, Brillo, pH)",
        "medicion_continua": "Variables con Medición Continua (temperaturas, flujos, presiones, consistencias)"
    }

    @classmethod
    def classify_features(cls, columns: List[str], metadata_tags: Dict[str, str] = None) -> pd.DataFrame:
        """Categorizes each feature according to chemical engineering and plant operation rules."""
        records = []
        for col in columns:
            col_lower = col.lower()
            tag = metadata_tags.get(col, "") if metadata_tags else ""

            # Check 1: Disabled / without measurement
            if "ce05_ip21_547fi1019" in col_lower or "unnamed" in col_lower:
                category = "sin_medicion"
                stage = "Inhabilitado"
            # Check 2: Fixed value / constant
            elif "agua de lavado" in col_lower or "cumple" in col_lower:
                category = "valor_fijo"
                stage = "Auxiliar"
            # Check 3: Control de Elementos (Torque, Motores, Válvulas, Nivel Control)
            elif any(k in col_lower for k in ["torque", "motor", "lic", "nic", "wic", "control cs"]):
                category = "control_elementos"
                stage = cls._detect_stage(col_lower)
            # Check 4: Criterio Experto (Kappa, Brillo, Blancura, pH)
            elif any(k in col_lower for k in ["kappa", "brillo", "blancura", "ph"]):
                category = "criterio_experto"
                stage = cls._detect_stage(col_lower)
            # Check 5: Continuous process measurements (Flujos, Presiones, Temperaturas, Consistencias, Factor dilución)
            else:
                category = "medicion_continua"
                stage = cls._detect_stage(col_lower)

            records.append({
                "Variable": col,
                "Tag": tag,
                "Categoria": cls.CATEGORY_MAPPING[category],
                "Categoria_Cod": category,
                "Etapa": stage
            })

        return pd.DataFrame(records)

    @staticmethod
    def _detect_stage(col_name: str) -> str:
        """Determines the chemical bleaching process stage based on variable nomenclature."""
        if "pre-blanq" in col_name or "preblanq" in col_name or "pre blanq" in col_name:
            return "Pre-Blanqueo"
        elif " d0" in col_name or "_d0" in col_name or "etapa d0" in col_name or "etapa do" in col_name:
            return "Etapa D0"
        elif "eop" in col_name or "e0p" in col_name or "op stage" in col_name or "op-reactor" in col_name:
            return "Etapa EOP"
        elif " d1" in col_name or "_d1" in col_name or "etapa d1" in col_name:
            return "Etapa D1"
        elif " d2" in col_name or "_d2" in col_name or "etapa d2" in col_name:
            return "Etapa D2 y Almacenamiento"
        elif "total" in col_name or "l1" in col_name or "concentracion" in col_name:
            return "Planta Química / Global"
        return "General"


class MulticollinearityAnalyzer:
    """Calculates Variance Inflation Factors (VIF) and condition indexes to diagnose multicollinearity."""

    @staticmethod
    def compute_vif(X: pd.DataFrame, sample_size: int = 2000) -> pd.DataFrame:
        """Computes VIF for a feature dataframe, sampling if necessary for high performance."""
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
        else:
            X_sample = X.copy()

        # Add constant for VIF intercept
        X_mat = X_sample.values
        vif_data = []

        for i, col_name in enumerate(X.columns):
            try:
                vif_val = variance_inflation_factor(X_mat, i)
            except Exception:
                vif_val = np.nan
            vif_data.append({"Variable": col_name, "VIF": vif_val})

        vif_df = pd.DataFrame(vif_data).sort_values(by="VIF", ascending=False)
        return vif_df
