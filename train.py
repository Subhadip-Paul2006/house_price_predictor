"""
train.py — Model training pipeline entry point.

Orchestrates the full training flow:
    load CSV → clean → engineer features → split → train → evaluate → persist

Supports multiple algorithms (linear_regression, random_forest).
Usage:
    python train.py
    python train.py --model random_forest
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import numpy as np

from evaluate import Evaluator
from model import ArtifactStore
from preprocess import CATEGORICAL_COLS, NUMERIC_COLS, HouseData, Preprocessor

# ── Logging setup ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 42

ALGORITHMS = {
    "linear_regression": {
        "import": "from sklearn.linear_model import LinearRegression",
        "class": "LinearRegression()",
        "display_name": "Linear Regression",
    },
    "random_forest": {
        "import": "from sklearn.ensemble import RandomForestRegressor",
        "class": "RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)",
        "display_name": "Random Forest",
    },
}


# ========================================================================
# ModelTrainer
# ========================================================================
class ModelTrainer:
    """Train a regression model from feature/target arrays."""

    def __init__(self, algorithm: str = "linear_regression", seed: int = SEED) -> None:
        self.algorithm = algorithm
        self.seed = seed
        self.model = self._build_model()

    def _build_model(self):
        """Instantiate the regressor based on algorithm name."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        if self.algorithm == "linear_regression":
            return LinearRegression()
        elif self.algorithm == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                random_state=self.seed,
            )
        else:
            raise ValueError(
                f"Unknown algorithm '{self.algorithm}'. "
                f"Choose from: {list(ALGORITHMS.keys())}"
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the model and return itself."""
        logger.info("Training %s on %d samples...", ALGORITHMS[self.algorithm]["display_name"], len(X_train))
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")
        return self

    def cross_validate(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> dict:
        """Run k-fold cross-validation and return mean/std scores."""
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(self.model, X, y, cv=k, scoring="r2")
        logger.info(
            "CV (%d-fold) R²: mean=%.4f  std=%.4f",
            k,
            scores.mean(),
            scores.std(),
        )
        return {"mean_r2": float(scores.mean()), "std_r2": float(scores.std())}


# ========================================================================
# Main pipeline
# ========================================================================
def run_pipeline(algorithm: str = "linear_regression") -> None:
    """Execute the full train → evaluate → save pipeline."""

    # 1. Load data
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading dataset")
    logger.info("=" * 60)
    house_data = HouseData()
    df = house_data.load()

    # 2. Clean data
    logger.info("STEP 2 — Cleaning data")
    df = Preprocessor.clean(df)

    # 3. Engineer features
    logger.info("STEP 3 — Engineering features")
    df = Preprocessor.engineer_features(df)

    # 4. Separate features and target
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df["Price"]

    # 5. Split
    logger.info("STEP 4 — Splitting train/test (80/20, seed=%d)", SEED)
    X_train, X_test, y_train, y_test = Preprocessor.split(X, y)

    # 6. Preprocess: fit on train, transform both
    logger.info("STEP 5 — Scaling & encoding")
    preprocessor = Preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # 7. Train
    logger.info("STEP 6 — Training model: %s", ALGORITHMS[algorithm]["display_name"])
    trainer = ModelTrainer(algorithm=algorithm)
    trainer.train(X_train_enc, y_train)

    # 8. Cross-validate
    logger.info("STEP 7 — Cross-validation")
    cv_scores = trainer.cross_validate(X_train_enc, y_train)

    # 9. Evaluate on test set
    logger.info("STEP 8 — Evaluating on test set")
    y_pred = trainer.model.predict(X_test_enc)
    metrics = Evaluator.report(y_test.values, y_pred)
    metrics["n_train"] = len(X_train)
    metrics["n_test"] = len(X_test)
    metrics["cv_mean_r2"] = cv_scores["mean_r2"]
    metrics["cv_std_r2"] = cv_scores["std_r2"]
    metrics["algorithm"] = algorithm
    metrics["timestamp"] = datetime.now().isoformat()

    # 10. Persist
    logger.info("STEP 9 — Saving artifacts")
    artifact_dir = ArtifactStore.save(
        model=trainer.model,
        preprocessor=preprocessor,
        metrics=metrics,
        algorithm_name=algorithm,
    )

    # 11. Print summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    print(f"\n📊 {ALGORITHMS[algorithm]['display_name']} — Results")
    print(f"   Training samples : {metrics['n_train']}")
    print(f"   Test samples     : {metrics['n_test']}")
    print(f"   MAE              : {metrics['mae']:.2f} Lakh")
    print(f"   RMSE             : {metrics['rmse']:.2f} Lakh")
    print(f"   R² (test)        : {metrics['r2']:.4f}")
    print(f"   R² (CV mean)     : {metrics['cv_mean_r2']:.4f} ± {metrics['cv_std_r2']:.4f}")
    print(f"   Residual Std     : {metrics['residual_std']:.2f} Lakh")
    print(f"   Artifact saved   : {artifact_dir}")
    print()


def train_all_models() -> None:
    """Train both Linear Regression and Random Forest, then compare."""
    results = {}

    for algo_key in ALGORITHMS:
        run_pipeline(algorithm=algo_key)
        # Load back metrics for comparison
        _, _, metrics = ArtifactStore.load(algorithm_name=algo_key)
        results[ALGORITHMS[algo_key]["display_name"]] = metrics

    # Comparison table
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(Evaluator.compare_models(results))


# ========================================================================
# CLI
# ========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the House Price Prediction model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(ALGORITHMS.keys()) + ["all"],
        default="all",
        help="Which model to train (default: all).",
    )
    args = parser.parse_args()

    if args.model == "all":
        train_all_models()
    else:
        run_pipeline(algorithm=args.model)


if __name__ == "__main__":
    main()
