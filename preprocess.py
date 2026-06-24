"""
preprocess.py — Data loading, cleaning, feature engineering, and transformation.

Classes:
    HouseData     — load & sanity-check a CSV dataset.
    Preprocessor  — clean, engineer features, encode, scale, and split.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

# ── Column groups ──────────────────────────────────────────────────────
NUMERIC_COLS = ["Area", "Bedrooms", "Bathrooms", "Age"]
CATEGORICAL_COLS = ["Location"]
TARGET_COL = "Price"


# ========================================================================
# HouseData
# ========================================================================
class HouseData:
    """Load and inspect a house-price CSV dataset."""

    def __init__(self, path: str = "data/house_data.csv") -> None:
        self.path = path
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Read the CSV into a DataFrame and store it."""
        self.df = pd.read_csv(self.path)
        logger.info("Loaded %d rows × %d columns from %s", *self.df.shape, self.path)
        return self.df

    def info(self) -> None:
        """Print DataFrame info (dtypes, non-null counts)."""
        if self.df is None:
            self.load()
        self.df.info()

    def describe(self) -> pd.DataFrame:
        """Return summary statistics."""
        if self.df is None:
            self.load()
        return self.df.describe()


# ========================================================================
# Preprocessor
# ========================================================================
class Preprocessor:
    """Clean, engineer features, encode categoricals, scale numerics."""

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

        # Drop rows with missing target
        n_before = len(cleaned)
        cleaned = cleaned.dropna(subset=[TARGET_COL])
        n_dropped = n_before - len(cleaned)
        if n_dropped:
            logger.info("Dropped %d rows with missing '%s'", n_dropped, TARGET_COL)

        # Impute numeric columns with median
        for col in NUMERIC_COLS:
            if col in cleaned.columns and cleaned[col].isnull().any():
                median_val = cleaned[col].median()
                cleaned = cleaned.assign(**{col: cleaned[col].fillna(median_val)})
                logger.info("Imputed '%s' NaN → median %.2f", col, median_val)

        # Impute categorical columns with mode
        for col in CATEGORICAL_COLS:
            if col in cleaned.columns and cleaned[col].isnull().any():
                mode_val = cleaned[col].mode()[0]
                cleaned = cleaned.assign(**{col: cleaned[col].fillna(mode_val)})
                logger.info("Imputed '%s' NaN → mode '%s'", col, mode_val)

        return cleaned

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to the DataFrame."""
        df = df.copy()

        # Price per square foot (only if target exists — training time)
        if TARGET_COL in df.columns and "Area" in df.columns:
            df["Price_Per_SqFt"] = df[TARGET_COL] / df["Area"]

        # Age bucket for non-linear age effects
        if "Age" in df.columns:
            bins = [0, 5, 10, 20, 50]
            labels = ["New", "Modern", "Old", "Very_Old"]
            df["Age_Bucket"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True)

        # Bedrooms-to-bathrooms ratio
        if "Bedrooms" in df.columns and "Bathrooms" in df.columns:
            df["BedBath_Ratio"] = df["Bedrooms"] / df["Bathrooms"].replace(0, 1)

        return df

    @staticmethod
    def split(
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Train/test split with a fixed seed for reproducibility."""
        return train_test_split(X, y, test_size=test_size, random_state=seed)

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit scalers/encoders on training data and return transformed array."""
        self._build_transformer()
        X_encoded = self.column_transformer.fit_transform(X_train)
        self._fitted = True
        logger.info("ColumnTransformer fitted on %d training samples", len(X_train))
        return X_encoded

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using the already-fitted transformer."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted — call fit_transform() first.")
        return self.column_transformer.transform(X)

    # ── Internals ──────────────────────────────────────────────────────

    def _build_transformer(self) -> None:
        """Build a sklearn ColumnTransformer for numeric + categorical columns.

        Numeric: impute (median) → scale (StandardScaler)  [via Pipeline]
        Categorical: one-hot encode (ignore unknowns at inference time)
        """
        from sklearn.pipeline import Pipeline

        self.column_transformer = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),  # safety net
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
