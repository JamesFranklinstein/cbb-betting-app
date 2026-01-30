"""
ML Predictor

Machine learning model for college basketball game predictions.
Uses XGBoost for efficient and interpretable predictions.
Uses KenPom metrics, height data, and historical data to improve predictions.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .xgboost_model import XGBoostPredictor, FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from ML model prediction."""
    home_win_prob: float
    spread_pred: float
    total_pred: float
    confidence: float
    features_used: Dict[str, float]
    model_version: str
    model_type: str = "xgboost"


class MLPredictor:
    """
    Machine learning predictor for college basketball games.

    Uses a combination of:
    - KenPom efficiency metrics
    - Four Factors
    - Height differentials
    - Historical matchup data
    - Home court advantage modeling

    Uses XGBoost for all predictions.
    """

    MODEL_VERSION = "v3.0-xgboost"

    # Feature columns for the model
    FEATURE_COLUMNS = FEATURE_COLUMNS

    def __init__(self, model_dir: str = "./models"):
        """
        Initialize the ML predictor.

        Args:
            model_dir: Directory for model storage
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # XGBoost predictor
        self.xgboost_predictor: Optional[XGBoostPredictor] = None

        # Training metrics
        self.training_metrics: Dict[str, float] = {}

        # Load models if available
        self._load_models()

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready for predictions."""
        return self.xgboost_predictor is not None and self.xgboost_predictor.is_fitted

    @property
    def active_model_type(self) -> str:
        """Get the type of currently active model."""
        return "xgboost" if self.is_ready else "none"

    # ==================== PREDICTION ====================

    def predict(
        self,
        home_team_stats: Dict[str, float],
        away_team_stats: Dict[str, float],
        neutral_site: bool = False
    ) -> PredictionResult:
        """
        Make a prediction for a game.

        Args:
            home_team_stats: KenPom stats for home team
            away_team_stats: KenPom stats for away team
            neutral_site: Whether game is at neutral site

        Returns:
            PredictionResult with probabilities and predictions
        """
        if not self.is_ready:
            raise ValueError("No model available. Call train() first or load a trained model.")

        # Create feature vector
        features = self._create_features(home_team_stats, away_team_stats, neutral_site)

        # Make prediction using XGBoost
        preds = self.xgboost_predictor.predict_single(features)

        # Calculate confidence
        confidence = self._calculate_confidence(preds['home_win_prob'], features)

        return PredictionResult(
            home_win_prob=preds['home_win_prob'],
            spread_pred=preds['spread'],
            total_pred=preds['total'],
            confidence=confidence,
            features_used=features,
            model_version=self.MODEL_VERSION,
            model_type="xgboost"
        )

    def predict_from_kenpom(
        self,
        home_rating: Dict[str, Any],
        away_rating: Dict[str, Any],
        neutral_site: bool = False,
        home_height_data: Dict[str, Any] = None,
        away_height_data: Dict[str, Any] = None
    ) -> PredictionResult:
        """
        Make prediction directly from KenPom API response data.

        Args:
            home_rating: KenPom rating response for home team
            away_rating: KenPom rating response for away team
            neutral_site: Whether game is at neutral site
            home_height_data: Optional height data from KenPom height endpoint
            away_height_data: Optional height data from KenPom height endpoint

        Returns:
            PredictionResult
        """
        home_stats = self._kenpom_to_features(home_rating, home_height_data)
        away_stats = self._kenpom_to_features(away_rating, away_height_data)
        return self.predict(home_stats, away_stats, neutral_site)

    # ==================== TRAINING ====================

    def train(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame = None,
        target_col: str = "home_won",
        spread_col: str = "actual_spread",
        total_col: str = "actual_total"
    ) -> Dict[str, float]:
        """
        Train the XGBoost ML models on historical data.

        Args:
            training_data: DataFrame with features and outcomes
            validation_data: Optional validation DataFrame
            target_col: Column name for win/loss outcome
            spread_col: Column name for actual spread
            total_col: Column name for actual total

        Returns:
            Dictionary of training metrics
        """
        # Initialize XGBoost predictor
        self.xgboost_predictor = XGBoostPredictor(model_dir=self.model_dir)

        # Train the model
        metrics = self.xgboost_predictor.train(
            train_df=training_data,
            val_df=validation_data,
            calibrate=True
        )

        # Save the model
        version = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        self.xgboost_predictor.save(version)

        # Flatten metrics for return
        self.training_metrics = {
            "accuracy": metrics["win"]["accuracy"],
            "brier_score": metrics["win"]["brier_score"],
            "log_loss": metrics["win"]["log_loss"],
            "spread_mae": metrics["spread"]["mae"],
            "spread_rmse": metrics["spread"]["rmse"],
            "total_mae": metrics["total"]["mae"],
            "total_rmse": metrics["total"]["rmse"],
        }

        logger.info(f"XGBoost model trained: {self.training_metrics}")
        return self.training_metrics

    # ==================== FEATURE ENGINEERING ====================

    def _create_features(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float],
        neutral_site: bool
    ) -> Dict[str, float]:
        """Create feature dictionary from team stats including height features."""
        features = {}

        # Calculate differentials (home - away)
        stat_pairs = [
            ("adj_em", "adj_em_diff"),
            ("adj_oe", "adj_oe_diff"),
            ("adj_de", "adj_de_diff"),
            ("adj_tempo", "adj_tempo_diff"),
            ("efg_pct", "efg_pct_diff"),
            ("to_pct", "to_pct_diff"),
            ("or_pct", "or_pct_diff"),
            ("ft_rate", "ft_rate_diff"),
            ("d_efg_pct", "d_efg_pct_diff"),
            ("d_to_pct", "d_to_pct_diff"),
            ("d_or_pct", "d_or_pct_diff"),
            ("d_ft_rate", "d_ft_rate_diff"),
            ("sos", "sos_diff"),
            ("luck", "luck_diff"),
            ("rank", "rank_diff"),
        ]

        for stat_key, feature_name in stat_pairs:
            home_val = home_stats.get(stat_key, 0)
            away_val = away_stats.get(stat_key, 0)
            features[feature_name] = home_val - away_val

        # Home advantage
        features["home_advantage"] = 0.0 if neutral_site else 1.0

        # Win streaks (if available)
        features["home_win_streak"] = home_stats.get("win_streak", 0)
        features["away_win_streak"] = away_stats.get("win_streak", 0)

        # Height features
        features.update(self._calculate_height_features(home_stats, away_stats))

        return features

    def _calculate_height_features(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate height-related features.

        Args:
            home_stats: Home team stats including avg_height, continuity
            away_stats: Away team stats

        Returns:
            Dict with height_diff, effective_height_diff, height_vs_tempo
        """
        # Get height values (default to 0 if not available)
        home_height = home_stats.get("avg_height", 0)
        away_height = away_stats.get("avg_height", 0)

        # 1. Raw height differential (in inches)
        height_diff = home_height - away_height

        # 2. Effective height weighted by roster continuity
        home_continuity = home_stats.get("continuity", 50) / 100.0
        away_continuity = away_stats.get("continuity", 50) / 100.0

        home_effective = home_height * home_continuity if home_height > 0 else 0
        away_effective = away_height * away_continuity if away_height > 0 else 0
        effective_height_diff = home_effective - away_effective

        # 3. Height-tempo interaction
        tempo_diff = home_stats.get("adj_tempo", 0) - away_stats.get("adj_tempo", 0)
        height_vs_tempo = height_diff * (-tempo_diff / 10.0) if height_diff != 0 else 0

        return {
            "height_diff": height_diff,
            "effective_height_diff": effective_height_diff,
            "height_vs_tempo": height_vs_tempo,
        }

    def _get_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dictionary to vector in correct order."""
        return [features.get(col, 0.0) for col in self.FEATURE_COLUMNS]

    def _kenpom_to_features(
        self,
        rating: Dict[str, Any],
        height_data: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Convert KenPom API response to feature dictionary.

        Args:
            rating: KenPom ratings response
            height_data: Optional KenPom height endpoint response

        Returns:
            Dict of feature values
        """
        features = {
            "adj_em": rating.get("AdjEM", 0),
            "adj_oe": rating.get("AdjOE", 0),
            "adj_de": rating.get("AdjDE", 0),
            "adj_tempo": rating.get("AdjTempo", 0),
            "efg_pct": rating.get("eFG_Pct", 0),
            "to_pct": rating.get("TO_Pct", 0),
            "or_pct": rating.get("OR_Pct", 0),
            "ft_rate": rating.get("FT_Rate", 0),
            "d_efg_pct": rating.get("DeFG_Pct", 0),
            "d_to_pct": rating.get("DTO_Pct", 0),
            "d_or_pct": rating.get("DOR_Pct", 0),
            "d_ft_rate": rating.get("DFT_Rate", 0),
            "sos": rating.get("SOS", 0),
            "luck": rating.get("Luck", 0),
            "rank": rating.get("RankAdjEM", 100),
        }

        # Add height data if available
        if height_data:
            features["avg_height"] = height_data.get("AvgHgt", 0)
            features["continuity"] = height_data.get("Continuity", 50)
            features["experience"] = height_data.get("Experience", 0)

        return features

    def _calculate_confidence(
        self,
        win_prob: float,
        features: Dict[str, float]
    ) -> float:
        """
        Calculate confidence in prediction.

        Higher confidence when:
        - Win probability is more extreme
        - Efficiency margin difference is large
        - Teams are well-separated in rankings
        """
        # Base confidence from win probability extremity
        prob_confidence = abs(win_prob - 0.5) * 2

        # Adjust based on efficiency margin difference
        em_diff = abs(features.get("adj_em_diff", 0))
        em_confidence = min(em_diff / 20, 1.0)

        # Combine
        confidence = (prob_confidence * 0.6) + (em_confidence * 0.4)

        return min(confidence, 1.0)

    # ==================== MODEL PERSISTENCE ====================

    def _load_models(self):
        """Load trained XGBoost model from disk."""
        try:
            self.xgboost_predictor = XGBoostPredictor(model_dir=self.model_dir)
            loaded = self.xgboost_predictor.load()

            if loaded:
                logger.info(f"Loaded XGBoost model from {self.model_dir}")
                if self.xgboost_predictor.metadata:
                    self.training_metrics = self.xgboost_predictor.metadata.get("metrics", {})
            else:
                logger.info("No XGBoost model found, will need to train")
                self.xgboost_predictor = None
        except Exception as e:
            logger.warning(f"Could not load XGBoost model: {e}")
            self.xgboost_predictor = None

    def load_model(self, model_path: str) -> None:
        """
        Load a specific XGBoost model.

        Args:
            model_path: Path to the model directory
        """
        self.xgboost_predictor = XGBoostPredictor(model_dir=self.model_dir)
        loaded = self.xgboost_predictor.load(model_path)

        if not loaded:
            raise ValueError(f"Failed to load model from {model_path}")

        logger.info(f"Loaded XGBoost model from {model_path}")

    # ==================== EVALUATION ====================

    def backtest(
        self,
        historical_data: pd.DataFrame,
        odds_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """
        Backtest model performance on historical data.

        Args:
            historical_data: DataFrame with features and outcomes
            odds_data: Optional DataFrame with historical odds

        Returns:
            Dictionary of backtest metrics
        """
        if not self.is_ready:
            raise ValueError("No model trained")

        # Get predictions
        win_probs, spreads, totals = self.xgboost_predictor.predict(historical_data)
        y_pred = (win_probs > 0.5).astype(int)
        y_true = historical_data["home_won"].values

        from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, np.clip(win_probs, 1e-7, 1-1e-7))),
            "brier_score": float(brier_score_loss(y_true, win_probs)),
            "n_games": len(y_true),
            "model_type": "xgboost"
        }

        # If we have odds, calculate ROI
        if odds_data is not None and "closing_odds" in odds_data.columns:
            roi = self._calculate_backtest_roi(win_probs, y_true, odds_data)
            results.update(roi)

        return results

    def _calculate_backtest_roi(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        odds_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate ROI from backtesting against historical odds."""
        min_edge = 0.05
        total_bets = 0
        total_profit = 0.0

        for i, (pred, actual) in enumerate(zip(predictions, actuals)):
            if i >= len(odds_data):
                break

            row = odds_data.iloc[i]
            implied_prob = row.get("implied_prob", 0.5)
            decimal_odds = row.get("decimal_odds", 2.0)

            edge = pred - implied_prob
            if abs(edge) >= min_edge:
                bet_on_home = edge > 0
                total_bets += 1

                if bet_on_home == bool(actual):
                    total_profit += decimal_odds - 1
                else:
                    total_profit -= 1

        return {
            "total_bets": total_bets,
            "total_profit": float(total_profit),
            "roi": float(total_profit / total_bets) if total_bets > 0 else 0.0
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Returns:
            Dict mapping feature names to importance values
        """
        if not self.is_ready:
            return {}

        importance = self.xgboost_predictor.get_feature_importance()
        # Return the win model's feature importance as the primary
        return importance.get("win", {})
