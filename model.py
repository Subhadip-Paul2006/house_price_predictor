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
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = "models"


# ========================================================================
# ArtifactStore
# ========================================================================
class ArtifactStore:
    """Save, load, and version trained model artifacts."""

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    @staticmethod
    def save(
        model: Any,
        preprocessor: Any,
        metrics: Dict[str, float],
        algorithm_name: str = "linear_regression",
        run_id: Optional[str] = None,
    ) -> str:
        """Persist model, preprocessor, and metrics to disk.

        Returns the artifact directory path.
        """
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        artifact_dir = os.path.join(MODEL_DIR, algorithm_name, run_id)
        os.makedirs(artifact_dir, exist_ok=True)

        # Save model + preprocessor as a single joblib bundle
        bundle_path = os.path.join(artifact_dir, "model.joblib")
        joblib.dump({"model": model, "preprocessor": preprocessor}, bundle_path)

        # Save metrics as JSON
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
        """Load the latest (or specified) artifact bundle.

        Returns (model, preprocessor, metrics_dict).
        """
        algo_dir = os.path.join(MODEL_DIR, algorithm_name)

        if run_id is None:
            # Pick the most recent run
            if not os.path.isdir(algo_dir):
                raise FileNotFoundError(
                    f"No artifacts found for '{algorithm_name}'. Run train.py first."
                )
            run_dirs = sorted(
                d for d in os.listdir(algo_dir)
                if os.path.isdir(os.path.join(algo_dir, d))
            )
            if not run_dirs:
                raise FileNotFoundError(
                    f"No run directories in '{algo_dir}'. Run train.py first."
                )
            run_id = run_dirs[-1]  # latest by name (timestamp-sorted)

        artifact_dir = os.path.join(algo_dir, run_id)
        bundle_path = os.path.join(artifact_dir, "model.joblib")
        metrics_path = os.path.join(artifact_dir, "metrics.json")

        if not os.path.exists(bundle_path):
            raise FileNotFoundError(f"Bundle not found at {bundle_path}")

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
        """Return the latest run-id (version tag) for the given algorithm."""
        algo_dir = os.path.join(MODEL_DIR, algorithm_name)
        if not os.path.isdir(algo_dir):
            return "none"
        run_dirs = sorted(
            d for d in os.listdir(algo_dir)
            if os.path.isdir(os.path.join(algo_dir, d))
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
        """Transform a feature dict and return the predicted price (Lakhs)."""
        X = self._features_to_dataframe(features)
        X_encoded = self.preprocessor.transform(X)
        price = float(self.model.predict(X_encoded)[0])
        logger.info("Predicted price: %.2f L for %s", price, features)
        return price

    def confidence(self, features: Dict[str, Any]) -> Tuple[float, float, float]:
        """Return (price, lower_bound, upper_bound) with 95 % confidence band."""
        price = self.predict(features)
        margin = 1.96 * self.residual_std   # 95 % CI
        return price, max(price - margin, 0.0), price + margin

    def _features_to_dataframe(self, features: Dict[str, Any]) -> "pd.DataFrame":
        """Convert a flat dict into a 1-row DataFrame matching training schema."""
        import pandas as pd

        # Ensure all expected columns are present
        row = {}
        for col in ["Area", "Bedrooms", "Bathrooms", "Age", "Location"]:
            row[col] = features.get(col, 0)
        return pd.DataFrame([row])
