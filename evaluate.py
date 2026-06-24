"""
evaluate.py — Model evaluation metrics (MAE, MSE, RMSE, R², full report).

Classes:
    Evaluator — static methods to compute regression metrics.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class Evaluator:
    """Compute and report regression evaluation metrics."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(mean_squared_error(y_true, y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² (coefficient of determination)."""
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def residual_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Standard deviation of residuals (used for confidence intervals)."""
        residuals = y_true - y_pred
        return float(np.std(residuals, ddof=1))

    @staticmethod
    def report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Return a dict with all metrics."""
        metrics = {
            "mae": Evaluator.mae(y_true, y_pred),
            "mse": Evaluator.mse(y_true, y_pred),
            "rmse": Evaluator.rmse(y_true, y_pred),
            "r2": Evaluator.r2(y_true, y_pred),
            "residual_std": Evaluator.residual_std(y_true, y_pred),
        }
        logger.info(
            "Evaluation — MAE: %.2f  RMSE: %.2f  R²: %.4f  Residual Std: %.2f",
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            metrics["residual_std"],
        )
        return metrics

    @staticmethod
    def compare_models(
        results: Dict[str, Dict[str, float]],
    ) -> str:
        """Pretty-print a comparison table of multiple models."""
        lines = []
        lines.append(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Res Std':>8}")
        lines.append("-" * 56)
        for name, m in results.items():
            lines.append(
                f"{name:<20} {m['mae']:>8.2f} {m['rmse']:>8.2f} "
                f"{m['r2']:>8.4f} {m['residual_std']:>8.2f}"
            )
        return "\n".join(lines)
