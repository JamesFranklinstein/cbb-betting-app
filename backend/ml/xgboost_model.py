"""
XGBoost Model for CBB Predictions

Replaces PyTorch with XGBoost for better interpretability and performance
on tabular data. Uses separate models for each prediction task:
- Win probability (classification with calibration)
- Spread prediction (regression)
- Total prediction (regression)

Version: 2.0.0 - Fixed is_fitted check during training
"""

import logging
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    brier_score_loss, log_loss, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

logger = logging.getLogger(__name__)


# Feature columns - same as PyTorch model for compatibility
FEATURE_COLUMNS = [
    # Efficiency differentials (18)
    "adj_em_diff", "adj_oe_diff", "adj_de_diff", "adj_tempo_diff",
    "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
    "d_efg_pct_diff", "d_to_pct_diff", "d_or_pct_diff", "d_ft_rate_diff",
    "sos_diff", "luck_diff", "home_advantage", "rank_diff",
    "home_win_streak", "away_win_streak",
    # Height features (3)
    "height_diff", "effective_height_diff", "height_vs_tempo",
]

# Target columns
TARGET_WIN = "home_won"
TARGET_SPREAD = "actual_spread"
TARGET_TOTAL = "actual_total"


class XGBoostPredictor:
    """
    XGBoost-based predictor for CBB games.

    Uses three separate models:
    - Win probability classifier with probability calibration
    - Spread regression model
    - Total regression model
    """

    def __init__(self, model_dir: str = "./ml_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.win_model: Optional[xgb.XGBClassifier] = None
        self.spread_model: Optional[xgb.XGBRegressor] = None
        self.total_model: Optional[xgb.XGBRegressor] = None

        # Calibrated classifier for better probability estimates
        self.calibrated_win_model: Optional[CalibratedClassifierCV] = None

        # Feature scaler
        self.scaler: Optional[StandardScaler] = None

        # Training metadata
        self.metadata: Dict[str, Any] = {}
        self.is_fitted = False

    def _get_default_params(self, task: str) -> Dict[str, Any]:
        """Get default XGBoost parameters for a task."""
        base_params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,  # L1 regularization
            "reg_lambda": 1.0,  # L2 regularization
            "random_state": 42,
            "n_jobs": -1,
        }

        if task == "classification":
            base_params["objective"] = "binary:logistic"
            base_params["eval_metric"] = "logloss"
        else:
            base_params["objective"] = "reg:squarederror"
            base_params["eval_metric"] = "rmse"

        return base_params

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from dataframe.

        Args:
            df: DataFrame with feature columns

        Returns:
            Numpy array of features
        """
        # Check for missing columns
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            logger.warning(f"Missing feature columns: {missing}")
            # Fill missing with zeros
            for col in missing:
                df[col] = 0.0

        X = df[FEATURE_COLUMNS].values.astype(np.float32)

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        win_params: Optional[Dict] = None,
        spread_params: Optional[Dict] = None,
        total_params: Optional[Dict] = None,
        calibrate: bool = True,
    ) -> Dict[str, Any]:
        """
        Train all three models.

        Args:
            train_df: Training data
            val_df: Validation data (optional, will split from train if not provided)
            win_params: XGBoost params for win model
            spread_params: XGBoost params for spread model
            total_params: XGBoost params for total model
            calibrate: Whether to calibrate win probability

        Returns:
            Dict with training metrics
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        logger.info(f"[XGBOOST_V2] Training XGBoost models on {len(train_df)} samples")

        # Split if no validation set provided
        if val_df is None:
            train_df, val_df = train_test_split(
                train_df, test_size=0.2, random_state=42
            )
            logger.info(f"Split: {len(train_df)} train, {len(val_df)} val")

        # Prepare features
        X_train = self.prepare_features(train_df)
        X_val = self.prepare_features(val_df)

        # Fit scaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Prepare targets
        y_train_win = train_df[TARGET_WIN].values.astype(int)
        y_val_win = val_df[TARGET_WIN].values.astype(int)

        y_train_spread = train_df[TARGET_SPREAD].values.astype(np.float32)
        y_val_spread = val_df[TARGET_SPREAD].values.astype(np.float32)

        y_train_total = train_df[TARGET_TOTAL].values.astype(np.float32)
        y_val_total = val_df[TARGET_TOTAL].values.astype(np.float32)

        metrics = {}

        # Train win probability model
        logger.info("Training win probability model...")
        win_params = win_params or self._get_default_params("classification")
        self.win_model = xgb.XGBClassifier(**win_params)
        self.win_model.fit(
            X_train_scaled, y_train_win,
            eval_set=[(X_val_scaled, y_val_win)],
            verbose=False
        )

        # Calibrate probabilities
        if calibrate:
            logger.info("Calibrating win probabilities...")
            self.calibrated_win_model = CalibratedClassifierCV(
                self.win_model, method='isotonic', cv='prefit'
            )
            self.calibrated_win_model.fit(X_val_scaled, y_val_win)

        # Set is_fitted before evaluation so predict methods work
        self.is_fitted = True
        logger.info(f"Set is_fitted = {self.is_fitted}")

        # Evaluate win model
        logger.info("Evaluating win model...")
        win_probs = self.predict_win_prob(X_val_scaled, already_scaled=True)
        metrics["win"] = {
            "accuracy": accuracy_score(y_val_win, (win_probs > 0.5).astype(int)),
            "brier_score": brier_score_loss(y_val_win, win_probs),
            "log_loss": log_loss(y_val_win, win_probs),
        }
        logger.info(f"Win model - Accuracy: {metrics['win']['accuracy']:.4f}, "
                   f"Brier: {metrics['win']['brier_score']:.4f}")

        # Train spread model
        logger.info("Training spread prediction model...")
        spread_params = spread_params or self._get_default_params("regression")
        self.spread_model = xgb.XGBRegressor(**spread_params)
        self.spread_model.fit(
            X_train_scaled, y_train_spread,
            eval_set=[(X_val_scaled, y_val_spread)],
            verbose=False
        )

        # Evaluate spread model
        spread_preds = self.spread_model.predict(X_val_scaled)
        metrics["spread"] = {
            "rmse": np.sqrt(mean_squared_error(y_val_spread, spread_preds)),
            "mae": mean_absolute_error(y_val_spread, spread_preds),
            "r2": r2_score(y_val_spread, spread_preds),
        }
        logger.info(f"Spread model - RMSE: {metrics['spread']['rmse']:.2f}, "
                   f"MAE: {metrics['spread']['mae']:.2f}")

        # Train total model
        logger.info("Training total prediction model...")
        total_params = total_params or self._get_default_params("regression")
        self.total_model = xgb.XGBRegressor(**total_params)
        self.total_model.fit(
            X_train_scaled, y_train_total,
            eval_set=[(X_val_scaled, y_val_total)],
            verbose=False
        )

        # Evaluate total model
        total_preds = self.total_model.predict(X_val_scaled)
        metrics["total"] = {
            "rmse": np.sqrt(mean_squared_error(y_val_total, total_preds)),
            "mae": mean_absolute_error(y_val_total, total_preds),
            "r2": r2_score(y_val_total, total_preds),
        }
        logger.info(f"Total model - RMSE: {metrics['total']['rmse']:.2f}, "
                   f"MAE: {metrics['total']['mae']:.2f}")

        # Convert metrics to native Python types for JSON serialization
        def to_native(obj):
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        metrics_native = to_native(metrics)

        # Store metadata
        self.metadata = {
            "trained_at": datetime.now().isoformat(),
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "feature_columns": FEATURE_COLUMNS,
            "metrics": metrics_native,
            "model_type": "xgboost",
        }

        return metrics

    def predict_win_prob(
        self,
        X: np.ndarray,
        already_scaled: bool = False
    ) -> np.ndarray:
        """Predict win probability."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if not already_scaled:
            X = self.scaler.transform(X)

        if self.calibrated_win_model is not None:
            return self.calibrated_win_model.predict_proba(X)[:, 1]
        else:
            return self.win_model.predict_proba(X)[:, 1]

    def predict_spread(
        self,
        X: np.ndarray,
        already_scaled: bool = False
    ) -> np.ndarray:
        """Predict spread (home - away)."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if not already_scaled:
            X = self.scaler.transform(X)

        return self.spread_model.predict(X)

    def predict_total(
        self,
        X: np.ndarray,
        already_scaled: bool = False
    ) -> np.ndarray:
        """Predict total points."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if not already_scaled:
            X = self.scaler.transform(X)

        return self.total_model.predict(X)

    def predict(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make all predictions for a dataframe.

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (win_probs, spreads, totals)
        """
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)

        win_probs = self.predict_win_prob(X_scaled, already_scaled=True)
        spreads = self.predict_spread(X_scaled, already_scaled=True)
        totals = self.predict_total(X_scaled, already_scaled=True)

        return win_probs, spreads, totals

    def predict_single(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Make predictions for a single game.

        Args:
            features: Dict of feature name -> value

        Returns:
            Dict with win_prob, spread, total predictions
        """
        df = pd.DataFrame([features])
        win_probs, spreads, totals = self.predict(df)

        return {
            "home_win_prob": float(win_probs[0]),
            "spread": float(spreads[0]),
            "total": float(totals[0]),
        }

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for all models."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        importance = {}

        for name, model in [
            ("win", self.win_model),
            ("spread", self.spread_model),
            ("total", self.total_model)
        ]:
            imp = model.feature_importances_
            importance[name] = {
                col: float(imp[i])
                for i, col in enumerate(FEATURE_COLUMNS)
            }
            # Sort by importance
            importance[name] = dict(
                sorted(importance[name].items(), key=lambda x: -x[1])
            )

        return importance

    def save(self, version: Optional[str] = None) -> str:
        """
        Save models to disk.

        Args:
            version: Optional version string

        Returns:
            Path to saved model directory
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")

        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        save_dir = self.model_dir / f"xgboost_v{version}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        self.win_model.save_model(str(save_dir / "win_model.json"))
        self.spread_model.save_model(str(save_dir / "spread_model.json"))
        self.total_model.save_model(str(save_dir / "total_model.json"))

        # Save calibrated model
        if self.calibrated_win_model is not None:
            with open(save_dir / "calibrated_win.pkl", "wb") as f:
                pickle.dump(self.calibrated_win_model, f)

        # Save scaler
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        self.metadata["version"] = version
        self.metadata["saved_at"] = datetime.now().isoformat()
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)

        # Also save as "best" model
        best_path = self.model_dir / "xgboost_best"
        if best_path.exists():
            import shutil
            shutil.rmtree(best_path)
        import shutil
        shutil.copytree(save_dir, best_path)

        logger.info(f"Model saved to {save_dir}")
        return str(save_dir)

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load models from disk.

        Args:
            path: Path to model directory. If None, loads "best" model.

        Returns:
            True if loaded successfully
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        if path is None:
            path = self.model_dir / "xgboost_best"
        else:
            path = Path(path)

        if not path.exists():
            logger.warning(f"Model path does not exist: {path}")
            return False

        try:
            # Load models
            self.win_model = xgb.XGBClassifier()
            self.win_model.load_model(str(path / "win_model.json"))

            self.spread_model = xgb.XGBRegressor()
            self.spread_model.load_model(str(path / "spread_model.json"))

            self.total_model = xgb.XGBRegressor()
            self.total_model.load_model(str(path / "total_model.json"))

            # Load calibrated model
            calibrated_path = path / "calibrated_win.pkl"
            if calibrated_path.exists():
                with open(calibrated_path, "rb") as f:
                    self.calibrated_win_model = pickle.load(f)

            # Load scaler
            with open(path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            # Load metadata
            with open(path / "metadata.json", "r") as f:
                self.metadata = json.load(f)

            self.is_fitted = True
            logger.info(f"Loaded XGBoost model from {path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


def train_xgboost_model(
    data_path: Optional[str] = None,
    output_dir: str = "./ml_models",
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Convenience function to train XGBoost model.

    Args:
        data_path: Path to training CSV. If None, loads from database.
        output_dir: Directory to save models
        test_size: Validation set size

    Returns:
        Training metrics and model path
    """
    logger.info("Starting XGBoost training...")

    # Load data
    if data_path:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
    else:
        # Load from database
        from ml.feedback_collector import FeedbackCollector
        from models.connection import SessionLocal

        session = SessionLocal()
        collector = FeedbackCollector(session)
        df = collector.get_training_dataframe()
        session.close()
        logger.info(f"Loaded {len(df)} samples from database")

    if len(df) < 100:
        raise ValueError(f"Not enough training data: {len(df)} samples")

    # Split data
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

    # Train model
    predictor = XGBoostPredictor(model_dir=output_dir)
    metrics = predictor.train(train_df, val_df)

    # Save model
    model_path = predictor.save()

    # Get feature importance
    importance = predictor.get_feature_importance()

    return {
        "metrics": metrics,
        "model_path": model_path,
        "feature_importance": importance,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = None

    results = train_xgboost_model(data_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nMetrics:")
    print(f"  Win - Accuracy: {results['metrics']['win']['accuracy']:.4f}, "
          f"Brier: {results['metrics']['win']['brier_score']:.4f}")
    print(f"  Spread - RMSE: {results['metrics']['spread']['rmse']:.2f}, "
          f"MAE: {results['metrics']['spread']['mae']:.2f}")
    print(f"  Total - RMSE: {results['metrics']['total']['rmse']:.2f}, "
          f"MAE: {results['metrics']['total']['mae']:.2f}")

    print(f"\nModel saved to: {results['model_path']}")

    print("\nTop 5 Features (Win Model):")
    for i, (feat, imp) in enumerate(list(results['feature_importance']['win'].items())[:5]):
        print(f"  {i+1}. {feat}: {imp:.4f}")
