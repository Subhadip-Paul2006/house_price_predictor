"""
model.py — Model artifact persistence and inference wrapper.

Classes:
    ArtifactStore — save / load / version model + preprocessor bundles.
    Predictor      — wraps a trained model + preprocessor for easy inference.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Bug Fix #20: Use absolute path for model directory
MODEL_DIR = str(Path(__file__).resolve().parent / "models")

# Feature columns expected during inference (must match preprocess.py)
ALL_FEATURE_COLS = [
    "Area", "Bedrooms", "Bathrooms", "Age",
    "Lot Area", "Overall Quality", "Overall Condition",
    "Garage Cars", "Garage Area", "Total Basement SF", "Fireplaces",
    "Location", "Neighborhood", "House Style", "Central Air", "Kitchen Quality",
]


# ========================================================================
# ArtifactStore
# ========================================================================
class ArtifactStore:
    """Save, load, and version trained model artifacts."""

    @staticmethod
    def save(
        model: Any,
        preprocessor: Any,
        metrics: Dict[str, float],
        algorithm_name: str = "linear_regression",
        run_id: Optional[str] = None,
    ) -> str:
        # Agar run_id nahi mila toh timestamp ko hi name bana denge folder ka
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        artifact_dir = os.path.join(MODEL_DIR, algorithm_name, run_id)
        os.makedirs(artifact_dir, exist_ok=True)

        # Model aur preprocessor ko single joblib dictionary bundle me pack karke write kar rahe hain
        bundle_path = os.path.join(artifact_dir, "model.joblib")
        joblib.dump({"model": model, "preprocessor": preprocessor}, bundle_path)

        # Metrics (R2, MAE, etc.) ko readable format me store karne ke liye
        metrics_path = os.path.join(artifact_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Saved artifact → %s", artifact_dir)
        return artifact_dir

    @staticmethod
    def load(
        algorithm_name: str = "linear_regression",
        run_id: Optional[str] = None,
    ) -> Tuple[Any, Any, Dict[str, float]]:
        # Train script ke artifacts load karne ka method
        algo_dir = os.path.join(MODEL_DIR, algorithm_name)

        if run_id is None:
            # Pura folder scan karke check karo kaunsa run complete hai (jahan model.joblib present ho)
            if not os.path.isdir(algo_dir):
                raise FileNotFoundError(
                    f"No artifacts found for '{algorithm_name}'. Run train.py first."
                )
            run_dirs = sorted(
                d for d in os.listdir(algo_dir)
                if os.path.isdir(os.path.join(algo_dir, d))
                and os.path.exists(os.path.join(algo_dir, d, "model.joblib"))
            )
            if not run_dirs:
                raise FileNotFoundError(
                    f"No valid run directories (containing model.joblib) in '{algo_dir}'. Run train.py first."
                )
            run_id = run_dirs[-1]  # Sabse latest complete run select kar rahe hain

        artifact_dir = os.path.join(algo_dir, run_id)
        bundle_path = os.path.join(artifact_dir, "model.joblib")
        metrics_path = os.path.join(artifact_dir, "metrics.json")

        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Bundle not found at {bundle_path}")

        # Model file parse kar rahe hain joblib ke through
        bundle = joblib.load(bundle_path)
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]

        metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)

        logger.info("Loaded artifact from %s", artifact_dir)
        return model, preprocessor, metrics

    @staticmethod
    def version(algorithm_name: str) -> str:
        # UI sidebar me version name print karne ke liye method
        algo_dir = os.path.join(MODEL_DIR, algorithm_name)
        if not os.path.isdir(algo_dir):
            return "none"
        run_dirs = sorted(
            d for d in os.listdir(algo_dir)
            if os.path.isdir(os.path.join(algo_dir, d))
            and os.path.exists(os.path.join(algo_dir, d, "model.joblib"))
        )
        return run_dirs[-1] if run_dirs else "none"


# ========================================================================
# Predictor
# ========================================================================
class Predictor:
    """High-level inference wrapper: model + preprocessor → price estimate."""

    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        residual_std: float = 0.0,
    ) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.residual_std = residual_std

    def predict(self, features: Dict[str, Any]) -> float:
        # Input parameters ko dataframe me daalkar transform karenge aur model se predict karayenge
        X = self._features_to_dataframe(features)
        X_encoded = self.preprocessor.transform(X)
        # Bug Fix #26: Clamp negative predictions to 0
        price = float(np.maximum(self.model.predict(X_encoded)[0], 0.0))
        logger.info("Predicted price: %.2f L for %s", price, features)
        return price

    def confidence(self, features: Dict[str, Any]) -> Tuple[float, float, float]:
        # Price estimate ke sath safe limits (Confidence interval) nikal rahe hain
        price = self.predict(features)
        margin = 1.96 * self.residual_std   # 95% Confidence Band
        return price, max(price - margin, 0.0), price + margin

    def _features_to_dataframe(self, features: Dict[str, Any]) -> "pd.DataFrame":
        # Streamlit input widgets ki raw dict values ko dataframe columns me format karne ke liye
        # Bug Fix #13: Uses ALL_FEATURE_COLS (single source of truth) instead of hardcoded list
        import pandas as pd

        row = {col: features.get(col, 0) for col in ALL_FEATURE_COLS}
        return pd.DataFrame([row])
