"""
ML Predictor

Machine learning model for college basketball game predictions.
Uses KenPom metrics and historical data to improve predictions.
"""

import os
import json
import pickle
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import joblib


@dataclass
class PredictionResult:
    """Result from ML model prediction."""
    home_win_prob: float
    spread_pred: float
    total_pred: float
    confidence: float
    features_used: Dict[str, float]
    model_version: str


class MLPredictor:
    """
    Machine learning predictor for college basketball games.
    
    Uses a combination of:
    - KenPom efficiency metrics
    - Four Factors
    - Historical matchup data
    - Home court advantage modeling
    """
    
    MODEL_VERSION = "v1.0"
    
    # Feature columns for the model
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
    ]
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Models
        self.win_prob_model: Optional[CalibratedClassifierCV] = None
        self.spread_model: Optional[RandomForestRegressor] = None
        self.total_model: Optional[RandomForestRegressor] = None
        
        # Feature scaler
        self.scaler: Optional[StandardScaler] = None
        
        # Training metrics
        self.training_metrics: Dict[str, float] = {}
        
        # Load models if available
        self._load_models()
    
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
        if self.win_prob_model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Create feature vector
        features = self._create_features(home_team_stats, away_team_stats, neutral_site)
        
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
        
        # Calculate confidence based on how extreme the prediction is
        confidence = self._calculate_confidence(win_prob, features)
        
        return PredictionResult(
            home_win_prob=float(win_prob),
            spread_pred=float(spread_pred),
            total_pred=float(total_pred),
            confidence=float(confidence),
            features_used=features,
            model_version=self.MODEL_VERSION
        )
    
    def predict_from_kenpom(
        self,
        home_rating: Dict[str, Any],
        away_rating: Dict[str, Any],
        neutral_site: bool = False
    ) -> PredictionResult:
        """
        Make prediction directly from KenPom API response data.
        
        Args:
            home_rating: KenPom rating response for home team
            away_rating: KenPom rating response for away team
            neutral_site: Whether game is at neutral site
            
        Returns:
            PredictionResult
        """
        home_stats = self._kenpom_to_features(home_rating)
        away_stats = self._kenpom_to_features(away_rating)
        return self.predict(home_stats, away_stats, neutral_site)
    
    # ==================== TRAINING ====================
    
    def train(
        self,
        training_data: pd.DataFrame,
        target_col: str = "home_win",
        spread_col: str = "actual_spread",
        total_col: str = "actual_total"
    ) -> Dict[str, float]:
        """
        Train the ML models on historical data.
        
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
            win_metrics["spread_mae"] = np.mean(np.abs(spread_preds - y_val_s))
        
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
            win_metrics["total_mae"] = np.mean(np.abs(total_preds - y_val_t))
        
        self.training_metrics = win_metrics
        
        # Save models
        self._save_models()
        
        return win_metrics
    
    # ==================== FEATURE ENGINEERING ====================
    
    def _create_features(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float],
        neutral_site: bool
    ) -> Dict[str, float]:
        """Create feature dictionary from team stats."""
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
        
        return features
    
    def _get_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Convert feature dictionary to vector in correct order."""
        return [features.get(col, 0.0) for col in self.FEATURE_COLUMNS]
    
    def _kenpom_to_features(self, rating: Dict[str, Any]) -> Dict[str, float]:
        """Convert KenPom API response to feature dictionary."""
        return {
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
        """Save trained models to disk."""
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
        
        print(f"Models saved to {self.model_dir}")
    
    def _load_models(self):
        """Load trained models from disk."""
        win_path = os.path.join(self.model_dir, "win_prob_model.pkl")
        spread_path = os.path.join(self.model_dir, "spread_model.pkl")
        total_path = os.path.join(self.model_dir, "total_model.pkl")
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        
        if os.path.exists(win_path):
            self.win_prob_model = joblib.load(win_path)
            print("Loaded win probability model")
        
        if os.path.exists(spread_path):
            self.spread_model = joblib.load(spread_path)
            print("Loaded spread model")
        
        if os.path.exists(total_path):
            self.total_model = joblib.load(total_path)
            print("Loaded total model")
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("Loaded scaler")
        
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                self.training_metrics = json.load(f)
    
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
        if self.win_prob_model is None:
            raise ValueError("Model not trained")
        
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in historical_data.columns]
        X = historical_data[feature_cols].fillna(0)
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Get predictions
        y_pred_proba = self.win_prob_model.predict_proba(X)[:, 1]
        y_pred = self.win_prob_model.predict(X)
        y_true = historical_data["home_win"]
        
        results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "log_loss": log_loss(y_true, y_pred_proba),
            "brier_score": brier_score_loss(y_true, y_pred_proba),
            "n_games": len(y_true)
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
        # Simple flat betting on value (model prob > implied prob + edge)
        min_edge = 0.05
        total_bets = 0
        total_profit = 0
        
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
            "total_profit": total_profit,
            "roi": total_profit / total_bets if total_bets > 0 else 0
        }
