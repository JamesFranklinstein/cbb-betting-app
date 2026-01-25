"""
ML Predictor

Machine learning model for college basketball game predictions.
Supports both sklearn (legacy) and PyTorch models.
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import joblib

# PyTorch imports (optional - graceful fallback if not installed)
try:
    import torch
    from .pytorch_model import CBBPredictionNet
    from .trainer import CBBModelTrainer
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    CBBPredictionNet = None
    CBBModelTrainer = None

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
    model_type: str = "sklearn"  # 'sklearn' or 'pytorch'


class MLPredictor:
    """
    Machine learning predictor for college basketball games.

    Uses a combination of:
    - KenPom efficiency metrics
    - Four Factors
    - Height differentials (NEW)
    - Historical matchup data
    - Home court advantage modeling

    Supports both sklearn (legacy) and PyTorch neural network models.
    """

    MODEL_VERSION = "v2.0"

    # Feature columns for the model (updated with height and situational features)
    FEATURE_COLUMNS = [
        # Team efficiency differentials
        "adj_em_diff",      # Adjusted efficiency margin
        "adj_oe_diff",      # Offensive efficiency
        "adj_de_diff",      # Defensive efficiency
        "adj_tempo_diff",   # Tempo

        # Four Factors (offense)
        "efg_pct_diff",     # Effective FG%
        "to_pct_diff",      # Turnover %
        "or_pct_diff",      # Offensive rebound %
        "ft_rate_diff",     # Free throw rate

        # Four Factors (defense)
        "d_efg_pct_diff",
        "d_to_pct_diff",
        "d_or_pct_diff",
        "d_ft_rate_diff",

        # Other factors
        "sos_diff",         # Strength of schedule
        "luck_diff",        # KenPom luck rating
        "home_advantage",   # Binary: 1 if home, 0 if neutral

        # Rankings
        "rank_diff",        # Ranking differential

        # Recent form (if available)
        "home_win_streak",
        "away_win_streak",

        # Height features (with low weight)
        "height_diff",              # Average height differential (inches)
        "effective_height_diff",    # Height weighted by roster continuity
        "height_vs_tempo",          # Height-tempo interaction

        # NEW: Situational features
        "days_of_season",           # Days since Nov 1 (season start) - captures time of season
        "is_conference_game",       # Binary: 1 if conference game
        "is_same_conference",       # Binary: 1 if same conference (derived)
        "home_rest_days",           # Days since home team's last game
        "away_rest_days",           # Days since away team's last game
        "rest_advantage",           # home_rest - away_rest (positive = home rested more)
        "home_back_to_back",        # Binary: 1 if home team on back-to-back
        "away_back_to_back",        # Binary: 1 if away team on back-to-back
        "travel_distance",          # Estimated travel distance for away team (miles/100)

        # NEW: Conference strength adjustments
        "home_conf_strength",       # Home team's conference average AdjEM
        "away_conf_strength",       # Away team's conference average AdjEM
        "conf_strength_diff",       # Difference in conference strength
    ]

    # Indices of height features for special weighting
    HEIGHT_FEATURE_INDICES = (18, 19, 20)
    HEIGHT_WEIGHT = 0.33  # Height features get 1/3 the normal weight

    # NEW: Indices of situational features (low weight - supplementary signals)
    SITUATIONAL_FEATURE_INDICES = tuple(range(21, 33))  # Indices 21-32
    SITUATIONAL_WEIGHT = 0.5  # Situational features get 1/2 normal weight

    def __init__(self, model_dir: str = "./models", use_pytorch: bool = True):
        """
        Initialize the ML predictor.

        Args:
            model_dir: Directory for model storage
            use_pytorch: If True, prefer PyTorch model when available
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        self.use_pytorch = use_pytorch and PYTORCH_AVAILABLE

        # sklearn models (legacy)
        self.win_prob_model: Optional[CalibratedClassifierCV] = None
        self.spread_model: Optional[RandomForestRegressor] = None
        self.total_model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None

        # PyTorch model and trainer
        self.pytorch_model: Optional[CBBPredictionNet] = None
        self.pytorch_trainer: Optional[CBBModelTrainer] = None
        self.pytorch_model_path: Optional[str] = None

        # Training metrics
        self.training_metrics: Dict[str, float] = {}

        # Track which model is active
        self._active_model: str = "none"  # 'sklearn', 'pytorch', or 'none'

        # Load models if available
        self._load_models()

    @property
    def is_ready(self) -> bool:
        """Check if any model is loaded and ready for predictions."""
        return self._active_model != "none"

    @property
    def active_model_type(self) -> str:
        """Get the type of currently active model."""
        return self._active_model

    # ==================== PREDICTION ====================

    def predict(
        self,
        home_team_stats: Dict[str, float],
        away_team_stats: Dict[str, float],
        neutral_site: bool = False,
        game_date: datetime = None,
        situational: Dict[str, Any] = None
    ) -> PredictionResult:
        """
        Make a prediction for a game.

        Args:
            home_team_stats: KenPom stats for home team
            away_team_stats: KenPom stats for away team
            neutral_site: Whether game is at neutral site
            game_date: Optional game date for time-of-season features
            situational: Optional dict with rest days, travel, conference info

        Returns:
            PredictionResult with probabilities and predictions
        """
        # Create feature vector with new situational features
        features = self._create_features(
            home_team_stats, away_team_stats, neutral_site,
            game_date=game_date, situational=situational
        )

        # Use PyTorch if available and preferred
        if self.use_pytorch and self.pytorch_trainer is not None:
            return self._predict_pytorch(features)
        elif self.win_prob_model is not None:
            return self._predict_sklearn(features)
        else:
            raise ValueError("No model available. Call train() first or load a trained model.")

    def _predict_pytorch(self, features: Dict[str, float]) -> PredictionResult:
        """Make prediction using PyTorch model."""
        preds = self.pytorch_trainer.predict(features)

        # Calculate confidence
        confidence = self._calculate_confidence(preds['win_prob'], features)

        return PredictionResult(
            home_win_prob=preds['win_prob'],
            spread_pred=preds['spread'],
            total_pred=preds['total'],
            confidence=confidence,
            features_used=features,
            model_version=self.MODEL_VERSION,
            model_type="pytorch"
        )

    def _predict_sklearn(self, features: Dict[str, float]) -> PredictionResult:
        """Make prediction using sklearn model."""
        # Ensure we have all required features
        feature_vector = self._get_feature_vector(features)

        # Scale features
        if self.scaler is not None:
            feature_vector = self.scaler.transform([feature_vector])
        else:
            feature_vector = [feature_vector]

        # Make predictions
        win_prob = self.win_prob_model.predict_proba(feature_vector)[0][1]
        spread_pred = self.spread_model.predict(feature_vector)[0] if self.spread_model else features["adj_em_diff"]
        total_pred = self.total_model.predict(feature_vector)[0] if self.total_model else 140.0

        # Calculate confidence
        confidence = self._calculate_confidence(win_prob, features)

        return PredictionResult(
            home_win_prob=float(win_prob),
            spread_pred=float(spread_pred),
            total_pred=float(total_pred),
            confidence=float(confidence),
            features_used=features,
            model_version=self.MODEL_VERSION,
            model_type="sklearn"
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

    def train_pytorch(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame = None,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        patience: int = 15
    ) -> Dict[str, float]:
        """
        Train the PyTorch model.

        Args:
            training_data: DataFrame with features and outcomes
            validation_data: Optional validation DataFrame
            epochs: Maximum epochs
            batch_size: Batch size
            learning_rate: Learning rate
            patience: Early stopping patience

        Returns:
            Dictionary of training metrics
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")

        # Split data if no validation provided
        if validation_data is None:
            train_df, val_df = train_test_split(
                training_data, test_size=0.2, random_state=42
            )
        else:
            train_df = training_data
            val_df = validation_data

        # Initialize trainer
        self.pytorch_trainer = CBBModelTrainer(
            model_dir=self.model_dir,
            height_weight=self.HEIGHT_WEIGHT
        )

        # Train
        metrics = self.pytorch_trainer.train(
            train_df=train_df,
            val_df=val_df,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience
        )

        # Calibrate temperature
        self.pytorch_trainer.calibrate_temperature(val_df)

        # Save model
        version = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_pytorch"
        self.pytorch_model_path = self.pytorch_trainer.save_model(version)

        # Update state
        self.pytorch_model = self.pytorch_trainer.model
        self._active_model = "pytorch"
        self.training_metrics = metrics

        logger.info(f"PyTorch model trained: {metrics}")
        return metrics

    def train(
        self,
        training_data: pd.DataFrame,
        target_col: str = "home_won",
        spread_col: str = "actual_spread",
        total_col: str = "actual_total"
    ) -> Dict[str, float]:
        """
        Train the sklearn ML models on historical data.

        Args:
            training_data: DataFrame with features and outcomes
            target_col: Column name for win/loss outcome
            spread_col: Column name for actual spread
            total_col: Column name for actual total

        Returns:
            Dictionary of training metrics
        """
        # Prepare features
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in training_data.columns]
        X = training_data[feature_cols].fillna(0)
        y_win = training_data[target_col]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Apply reduced weight to height features
        for idx in self.HEIGHT_FEATURE_INDICES:
            if idx < X_scaled.shape[1]:
                X_scaled[:, idx] *= self.HEIGHT_WEIGHT

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_win, test_size=0.2, random_state=42
        )

        # Train win probability model (calibrated for good probabilities)
        base_clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.win_prob_model = CalibratedClassifierCV(base_clf, cv=5, method="isotonic")
        self.win_prob_model.fit(X_train, y_train)

        # Evaluate win probability model
        y_pred_proba = self.win_prob_model.predict_proba(X_val)[:, 1]
        y_pred = self.win_prob_model.predict(X_val)

        win_metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "log_loss": log_loss(y_val, y_pred_proba),
            "brier_score": brier_score_loss(y_val, y_pred_proba)
        }

        # Train spread model
        if spread_col in training_data.columns:
            y_spread = training_data[spread_col].fillna(0)
            X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
                X_scaled, y_spread, test_size=0.2, random_state=42
            )

            self.spread_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.spread_model.fit(X_train_s, y_train_s)

            spread_preds = self.spread_model.predict(X_val_s)
            win_metrics["spread_mae"] = float(np.mean(np.abs(spread_preds - y_val_s)))

        # Train total model
        if total_col in training_data.columns:
            y_total = training_data[total_col].fillna(140)
            X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(
                X_scaled, y_total, test_size=0.2, random_state=42
            )

            self.total_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
            self.total_model.fit(X_train_t, y_train_t)

            total_preds = self.total_model.predict(X_val_t)
            win_metrics["total_mae"] = float(np.mean(np.abs(total_preds - y_val_t)))

        self.training_metrics = win_metrics
        self._active_model = "sklearn"

        # Save models
        self._save_models()

        return win_metrics

    # ==================== FEATURE ENGINEERING ====================

    def _create_features(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float],
        neutral_site: bool,
        game_date: datetime = None,
        situational: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Create feature dictionary from team stats including height and situational features.

        Args:
            home_stats: KenPom stats for home team
            away_stats: KenPom stats for away team
            neutral_site: Whether game is at neutral site
            game_date: Optional game date for time-of-season features
            situational: Optional dict with rest days, travel, etc.
        """
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

        # NEW: Situational features
        features.update(self._calculate_situational_features(
            home_stats, away_stats, game_date, situational
        ))

        return features

    def _calculate_situational_features(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float],
        game_date: datetime = None,
        situational: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Calculate situational features for improved predictions.

        These features capture:
        - Time of season (early season ratings less reliable)
        - Conference game indicator
        - Rest days and back-to-back situations
        - Travel distance impact
        - Conference strength adjustments

        Args:
            home_stats: Home team stats including conference
            away_stats: Away team stats including conference
            game_date: Date of the game
            situational: Dict with rest_days, travel info, etc.

        Returns:
            Dict with situational feature values
        """
        situational = situational or {}
        features = {}

        # 1. Days of season (0 = Nov 1, captures early season uncertainty)
        if game_date:
            # Season typically starts Nov 1
            season_start = datetime(game_date.year if game_date.month >= 8 else game_date.year - 1, 11, 1, tzinfo=timezone.utc)
            if game_date.tzinfo is None:
                game_date = game_date.replace(tzinfo=timezone.utc)
            days_since_start = (game_date - season_start).days
            # Normalize: 0-150 days typical season -> 0-1 scale
            features["days_of_season"] = min(max(days_since_start, 0), 150) / 150.0
        else:
            features["days_of_season"] = 0.5  # Default to mid-season

        # 2. Conference game indicators
        home_conf = home_stats.get("conference", "")
        away_conf = away_stats.get("conference", "")
        is_same_conf = 1.0 if (home_conf and away_conf and home_conf == away_conf) else 0.0
        features["is_conference_game"] = situational.get("is_conference_game", is_same_conf)
        features["is_same_conference"] = is_same_conf

        # 3. Rest days and back-to-back
        home_rest = situational.get("home_rest_days", 3)  # Default 3 days
        away_rest = situational.get("away_rest_days", 3)
        features["home_rest_days"] = min(home_rest, 7) / 7.0  # Normalize, cap at 7
        features["away_rest_days"] = min(away_rest, 7) / 7.0
        features["rest_advantage"] = (home_rest - away_rest) / 7.0  # Positive = home more rested

        # Back-to-back detection (played yesterday)
        features["home_back_to_back"] = 1.0 if home_rest <= 1 else 0.0
        features["away_back_to_back"] = 1.0 if away_rest <= 1 else 0.0

        # 4. Travel distance (normalized by 1000 miles)
        # Large travel = fatigue for away team
        travel = situational.get("travel_distance", 0)
        features["travel_distance"] = min(travel, 3000) / 1000.0  # Cap at 3000 miles

        # 5. Conference strength adjustments
        # Average AdjEM for each conference (helps detect weak/strong conference games)
        home_conf_strength = home_stats.get("conf_adj_em", 0)
        away_conf_strength = away_stats.get("conf_adj_em", 0)
        features["home_conf_strength"] = home_conf_strength / 20.0  # Normalize ~(-10, +10) range
        features["away_conf_strength"] = away_conf_strength / 20.0
        features["conf_strength_diff"] = (home_conf_strength - away_conf_strength) / 20.0

        return features

    def _calculate_height_features(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate height-related features.

        These features have low weight (0.33x) as requested by user.

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
        # Teams with more returning players have more reliable height data
        home_continuity = home_stats.get("continuity", 50) / 100.0
        away_continuity = away_stats.get("continuity", 50) / 100.0

        home_effective = home_height * home_continuity if home_height > 0 else 0
        away_effective = away_height * away_continuity if away_height > 0 else 0
        effective_height_diff = home_effective - away_effective

        # 3. Height-tempo interaction
        # Height matters more in slower, half-court games
        # In fast-paced games, athleticism matters more than height
        tempo_diff = home_stats.get("adj_tempo", 0) - away_stats.get("adj_tempo", 0)
        # Negative tempo_diff scaled so slower games amplify height advantage
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
        em_confidence = min(em_diff / 20, 1.0)  # Max out at 20 point difference

        # Combine
        confidence = (prob_confidence * 0.6) + (em_confidence * 0.4)

        return min(confidence, 1.0)

    # ==================== MODEL PERSISTENCE ====================

    def _save_models(self):
        """Save trained sklearn models to disk."""
        if self.win_prob_model is not None:
            joblib.dump(
                self.win_prob_model,
                os.path.join(self.model_dir, "win_prob_model.pkl")
            )

        if self.spread_model is not None:
            joblib.dump(
                self.spread_model,
                os.path.join(self.model_dir, "spread_model.pkl")
            )

        if self.total_model is not None:
            joblib.dump(
                self.total_model,
                os.path.join(self.model_dir, "total_model.pkl")
            )

        if self.scaler is not None:
            joblib.dump(
                self.scaler,
                os.path.join(self.model_dir, "scaler.pkl")
            )

        # Save metrics
        with open(os.path.join(self.model_dir, "metrics.json"), "w") as f:
            json.dump(self.training_metrics, f, indent=2)

        logger.info(f"sklearn models saved to {self.model_dir}")

    def _load_models(self):
        """Load trained models from disk (sklearn and/or PyTorch)."""
        # Try to load sklearn models
        win_path = os.path.join(self.model_dir, "win_prob_model.pkl")
        spread_path = os.path.join(self.model_dir, "spread_model.pkl")
        total_path = os.path.join(self.model_dir, "total_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        metrics_path = os.path.join(self.model_dir, "metrics.json")

        sklearn_loaded = False
        if os.path.exists(win_path):
            self.win_prob_model = joblib.load(win_path)
            sklearn_loaded = True
            logger.info("Loaded sklearn win probability model")

        if os.path.exists(spread_path):
            self.spread_model = joblib.load(spread_path)
            logger.info("Loaded sklearn spread model")

        if os.path.exists(total_path):
            self.total_model = joblib.load(total_path)
            logger.info("Loaded sklearn total model")

        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded sklearn scaler")

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.training_metrics = json.load(f)

        # Try to load PyTorch model
        if PYTORCH_AVAILABLE:
            pytorch_models = [f for f in os.listdir(self.model_dir)
                             if f.startswith("cbb_model_") and f.endswith(".pt")]
            if pytorch_models:
                # Load the most recent model
                pytorch_models.sort(reverse=True)
                latest_model = pytorch_models[0]
                model_path = os.path.join(self.model_dir, latest_model)

                self.pytorch_trainer = CBBModelTrainer(
                    model_dir=self.model_dir,
                    height_weight=self.HEIGHT_WEIGHT
                )
                self.pytorch_trainer.load_model(model_path)
                self.pytorch_model = self.pytorch_trainer.model
                self.pytorch_model_path = model_path
                logger.info(f"Loaded PyTorch model: {latest_model}")

                # Prefer PyTorch if configured and available
                if self.use_pytorch:
                    self._active_model = "pytorch"
                    return

        if sklearn_loaded:
            self._active_model = "sklearn"

    def load_pytorch_model(self, model_path: str) -> None:
        """
        Load a specific PyTorch model.

        Args:
            model_path: Path to the .pt model file
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available")

        self.pytorch_trainer = CBBModelTrainer(
            model_dir=self.model_dir,
            height_weight=self.HEIGHT_WEIGHT
        )
        self.pytorch_trainer.load_model(model_path)
        self.pytorch_model = self.pytorch_trainer.model
        self.pytorch_model_path = model_path
        self._active_model = "pytorch"
        logger.info(f"Loaded PyTorch model from {model_path}")

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
        if self._active_model == "pytorch" and self.pytorch_trainer:
            return self._backtest_pytorch(historical_data, odds_data)
        elif self.win_prob_model is not None:
            return self._backtest_sklearn(historical_data, odds_data)
        else:
            raise ValueError("No model trained")

    def _backtest_sklearn(
        self,
        historical_data: pd.DataFrame,
        odds_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """Backtest sklearn model."""
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in historical_data.columns]
        X = historical_data[feature_cols].fillna(0)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Get predictions
        y_pred_proba = self.win_prob_model.predict_proba(X)[:, 1]
        y_pred = self.win_prob_model.predict(X)
        y_true = historical_data["home_won"]

        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, y_pred_proba)),
            "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
            "n_games": len(y_true),
            "model_type": "sklearn"
        }

        # If we have odds, calculate ROI
        if odds_data is not None and "closing_odds" in odds_data.columns:
            roi = self._calculate_backtest_roi(y_pred_proba, y_true.values, odds_data)
            results.update(roi)

        return results

    def _backtest_pytorch(
        self,
        historical_data: pd.DataFrame,
        odds_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """Backtest PyTorch model."""
        # Get predictions
        result_df = self.pytorch_trainer.predict_batch(historical_data)
        y_pred_proba = result_df['pred_win_prob'].values
        y_pred = (y_pred_proba > 0.5).astype(int)
        y_true = historical_data["home_won"].values

        results = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "log_loss": float(log_loss(y_true, np.clip(y_pred_proba, 1e-7, 1-1e-7))),
            "brier_score": float(brier_score_loss(y_true, y_pred_proba)),
            "n_games": len(y_true),
            "model_type": "pytorch"
        }

        # If we have odds, calculate ROI
        if odds_data is not None and "closing_odds" in odds_data.columns:
            roi = self._calculate_backtest_roi(y_pred_proba, y_true, odds_data)
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
        if self._active_model == "pytorch" and self.pytorch_model:
            return self.pytorch_model.get_feature_importance(self.FEATURE_COLUMNS)
        elif self.win_prob_model is not None:
            # For sklearn, use the base estimator's feature importance
            try:
                base = self.win_prob_model.calibrated_classifiers_[0].estimator
                if hasattr(base, 'feature_importances_'):
                    importance = base.feature_importances_
                    return {
                        name: float(imp)
                        for name, imp in zip(self.FEATURE_COLUMNS, importance)
                    }
            except Exception:
                pass

        return {}
