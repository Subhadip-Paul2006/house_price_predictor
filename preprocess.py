"""
preprocess.py — Data loading, cleaning, feature engineering, and transformation.

Classes:
    HouseData     — load & sanity-check a CSV dataset.
    Preprocessor  — clean, engineer features, encode, scale, and split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ── Column groups ──────────────────────────────────────────────────────
# Kaunse columns numeric hain, kaunse categorical hain, aur target kya hai, yahan define kiya hai
# Updated for 16-feature CSV (17 columns total including Price target)
NUMERIC_COLS = [
    "Area", "Bedrooms", "Bathrooms", "Age",
    "Lot Area", "Overall Quality", "Overall Condition",
    "Garage Cars", "Garage Area", "Total Basement SF", "Fireplaces",
]
CATEGORICAL_COLS = ["Location", "Neighborhood", "House Style", "Central Air", "Kitchen Quality"]
TARGET_COL = "Price"

# Complete ordered list of all feature columns (used for DataFrame construction)
ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Base path resolution — works regardless of current working directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


# ========================================================================
# HouseData
# ========================================================================
class HouseData:
    """Load and inspect a house-price CSV dataset."""

    def __init__(self, path: Optional[str] = None) -> None:
        # Bug Fix #20: Use absolute path so it works from any CWD
        self.path = str(path or (DATA_DIR / "house_data.csv"))
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        # CSV file load karke pandas dataframe me convert kar rahe hain
        self.df = pd.read_csv(self.path)
        logger.info("Loaded %d rows × %d columns from %s", *self.df.shape, self.path)
        return self.df

    def info(self) -> None:
        # DataFrame ka structural overview check karne ke liye
        if self.df is None:
            self.load()
        self.df.info()

    def describe(self) -> pd.DataFrame:
        # Pura statistics (mean, count, min, max) dekhne ke liye
        if self.df is None:
            self.load()
        return self.df.describe()


# ========================================================================
# Preprocessor
# ========================================================================
class Preprocessor:
    """Clean, encode categoricals, scale numerics."""

    def __init__(self) -> None:
        self.scaler: Optional[StandardScaler] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.column_transformer: Optional[ColumnTransformer] = None
        self._fitted: bool = False

    # ── Public API ──────────────────────────────────────────────────────

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values.

        - Numeric columns: impute with median.
        - Categorical columns: impute with mode.
        - Drop rows where the target (Price) is missing.
        """
        cleaned = df.copy()

        # Agar actual price hi nahi diya toh row delete kar do (warna model train kaise hoga?)
        n_before = len(cleaned)
        cleaned = cleaned.dropna(subset=[TARGET_COL])
        n_dropped = n_before - len(cleaned)
        if n_dropped:
            logger.info("Dropped %d rows with missing '%s'", n_dropped, TARGET_COL)

        # Numeric columns me jahan blank tha wahan median daal rahe hain
        for col in NUMERIC_COLS:
            if col in cleaned.columns and cleaned[col].isnull().any():
                median_val = cleaned[col].median()
                cleaned = cleaned.assign(**{col: cleaned[col].fillna(median_val)})
                logger.info("Imputed '%s' NaN → median %.2f", col, median_val)

        # Categorical location blank ho toh most common location (mode) fill kar rahe hain
        for col in CATEGORICAL_COLS:
            if col in cleaned.columns and cleaned[col].isnull().any():
                mode_val = cleaned[col].mode()[0]
                cleaned = cleaned.assign(**{col: cleaned[col].fillna(mode_val)})
                logger.info("Imputed '%s' NaN → mode '%s'", col, mode_val)

        return cleaned

    @staticmethod
    def split(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        # Data split kar rahe hain: 80% Training ke liye, 20% Testing ke liye
        return train_test_split(X, y, test_size=test_size, random_state=seed)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        # Preprocessing setup run karke fit kar rahe hain (sirf train set par)
        self._build_transformer()
        X_encoded = self.column_transformer.fit_transform(X_train)
        self._fitted = True
        logger.info("ColumnTransformer fitted on %d training samples", len(X_train))
        return X_encoded

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # Naye data (e.g., users input) ko model-ready input array me convert karne ke liye
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted — call fit_transform() first.")
        return self.column_transformer.transform(X)

    # ── Internals ──────────────────────────────────────────────────────

    def _build_transformer(self) -> None:
        # sklearn ka standard ColumnTransformer pipeline set kar rahe hain
        # 1. Numeric -> Standard Scaler lagao
        # 2. Categorical -> One Hot Encoding lagao

        self.column_transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    NUMERIC_COLS,
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_COLS,
                ),
            ],
            remainder="drop",
        )
