"""
Feedback Collector Service

Collects feedback by joining predictions with actual game results.
Builds training data for model retraining and tracks prediction accuracy.
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """
    Collects feedback by joining predictions with actual game results.
    Builds training data for model retraining.

    This service is the core of the feedback loop:
    1. Finds completed games with predictions
    2. Joins features + predictions + actual results
    3. Stores as training data for model improvement
    4. Tracks prediction accuracy over time
    """

    def __init__(self, session: Session):
        """
        Initialize feedback collector.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session

    def collect_completed_games(
        self,
        days_back: int = 7,
        season: int = None
    ) -> List[Dict[str, Any]]:
        """
        Collect completed games with predictions for training data.

        Args:
            days_back: Number of days to look back for completed games
            season: Optional season filter (e.g., 2025)

        Returns:
            List of game data dicts with predictions and results
        """
        # Import models here to avoid circular imports
        from models.database import Game, Prediction, Team, TeamRating

        cutoff = datetime.utcnow() - timedelta(days=days_back)

        # Build query for completed games
        query = self.session.query(Game).filter(
            and_(
                Game.is_completed == True,
                Game.scheduled_time >= cutoff,
                Game.home_score.isnot(None),
                Game.away_score.isnot(None)
            )
        )

        if season:
            query = query.filter(Game.season == season)

        games = query.all()
        logger.info(f"Found {len(games)} completed games to process")

        results = []
        for game in games:
            try:
                game_data = self._process_game(game)
                if game_data:
                    results.append(game_data)
            except Exception as e:
                logger.warning(f"Error processing game {game.id}: {e}")
                continue

        logger.info(f"Successfully processed {len(results)} games")
        return results

    def _process_game(self, game) -> Optional[Dict[str, Any]]:
        """
        Process a single completed game into training data format.

        Args:
            game: Game model instance

        Returns:
            Dict with game data or None if missing required data
        """
        from models.database import Prediction, TeamRating

        # Get predictions for this game
        predictions = self.session.query(Prediction).filter(
            Prediction.game_id == game.id
        ).all()

        if not predictions:
            logger.debug(f"No predictions found for game {game.id}")
            return None

        # Get team ratings at game time
        home_rating = self._get_team_rating(game.home_team_id, game.scheduled_time)
        away_rating = self._get_team_rating(game.away_team_id, game.scheduled_time)

        # Build feature dict from ratings
        features = self._build_features_from_ratings(home_rating, away_rating)

        # Find ML and KenPom predictions
        ml_pred = None
        kp_pred = None
        for pred in predictions:
            if pred.source and pred.source.startswith('ml_'):
                ml_pred = pred
            elif pred.source == 'kenpom':
                kp_pred = pred

        # Calculate actual results
        actual_spread = game.home_score - game.away_score
        actual_total = game.home_score + game.away_score
        home_won = game.home_score > game.away_score

        return {
            'game_id': game.id,
            'season': game.season,
            'game_date': game.scheduled_time,
            'home_team_id': game.home_team_id,
            'away_team_id': game.away_team_id,
            'features': features,

            # ML predictions (if available)
            'pred_home_win_prob': ml_pred.home_win_prob if ml_pred else None,
            'pred_spread': ml_pred.spread_pred if ml_pred else None,
            'pred_total': ml_pred.total_pred if ml_pred else None,
            'pred_model_version': ml_pred.source if ml_pred else None,

            # KenPom predictions
            'kenpom_home_win_prob': kp_pred.home_win_prob if kp_pred else None,
            'kenpom_spread': kp_pred.spread_pred if kp_pred else None,
            'kenpom_total': kp_pred.total_pred if kp_pred else None,

            # Actual results
            'actual_home_score': game.home_score,
            'actual_away_score': game.away_score,
            'actual_spread': actual_spread,
            'actual_total': actual_total,
            'home_won': home_won,
        }

    def _get_team_rating(
        self,
        team_id: int,
        game_date: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Get team rating closest to (but before) game date.

        Args:
            team_id: Team database ID
            game_date: Date of the game

        Returns:
            Dict with rating data or None
        """
        from models.database import TeamRating

        rating = self.session.query(TeamRating).filter(
            and_(
                TeamRating.team_id == team_id,
                TeamRating.date <= game_date
            )
        ).order_by(TeamRating.date.desc()).first()

        if rating:
            return {
                'adj_em': rating.adj_em or 0,
                'adj_oe': rating.adj_oe or 0,
                'adj_de': rating.adj_de or 0,
                'adj_tempo': rating.adj_tempo or 0,
                'efg_pct': rating.efg_pct or 0,
                'to_pct': rating.to_pct or 0,
                'or_pct': rating.or_pct or 0,
                'ft_rate': rating.ft_rate or 0,
                'd_efg_pct': rating.d_efg_pct or 0,
                'd_to_pct': rating.d_to_pct or 0,
                'd_or_pct': rating.d_or_pct or 0,
                'd_ft_rate': rating.d_ft_rate or 0,
                'sos': rating.sos or 0,
                'luck': rating.luck or 0,
                'rank': rating.rank_adj_em or 100,
            }
        return None

    def _build_features_from_ratings(
        self,
        home_rating: Optional[Dict[str, Any]],
        away_rating: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Build feature dictionary from team ratings.

        Args:
            home_rating: Home team rating dict
            away_rating: Away team rating dict

        Returns:
            Dict of feature differentials
        """
        if not home_rating:
            home_rating = {}
        if not away_rating:
            away_rating = {}

        features = {
            'adj_em_diff': home_rating.get('adj_em', 0) - away_rating.get('adj_em', 0),
            'adj_oe_diff': home_rating.get('adj_oe', 0) - away_rating.get('adj_oe', 0),
            'adj_de_diff': home_rating.get('adj_de', 0) - away_rating.get('adj_de', 0),
            'adj_tempo_diff': home_rating.get('adj_tempo', 0) - away_rating.get('adj_tempo', 0),
            'efg_pct_diff': home_rating.get('efg_pct', 0) - away_rating.get('efg_pct', 0),
            'to_pct_diff': home_rating.get('to_pct', 0) - away_rating.get('to_pct', 0),
            'or_pct_diff': home_rating.get('or_pct', 0) - away_rating.get('or_pct', 0),
            'ft_rate_diff': home_rating.get('ft_rate', 0) - away_rating.get('ft_rate', 0),
            'd_efg_pct_diff': home_rating.get('d_efg_pct', 0) - away_rating.get('d_efg_pct', 0),
            'd_to_pct_diff': home_rating.get('d_to_pct', 0) - away_rating.get('d_to_pct', 0),
            'd_or_pct_diff': home_rating.get('d_or_pct', 0) - away_rating.get('d_or_pct', 0),
            'd_ft_rate_diff': home_rating.get('d_ft_rate', 0) - away_rating.get('d_ft_rate', 0),
            'sos_diff': home_rating.get('sos', 0) - away_rating.get('sos', 0),
            'luck_diff': home_rating.get('luck', 0) - away_rating.get('luck', 0),
            'home_advantage': 1.0,  # TODO: detect neutral site from game data
            'rank_diff': home_rating.get('rank', 100) - away_rating.get('rank', 100),
            'home_win_streak': 0,  # TODO: calculate from game history
            'away_win_streak': 0,
            # Height features - placeholder, need height data
            'height_diff': 0.0,
            'effective_height_diff': 0.0,
            'height_vs_tempo': 0.0,
        }

        return features

    def store_training_data(
        self,
        game_data: Dict[str, Any]
    ) -> Any:
        """
        Store collected game data as training data record.

        Args:
            game_data: Dict from collect_completed_games()

        Returns:
            TrainingData model instance
        """
        from models.database import TrainingData

        # Check if already stored
        existing = self.session.query(TrainingData).filter(
            TrainingData.game_id == game_data['game_id']
        ).first()

        if existing:
            logger.debug(f"Training data already exists for game {game_data['game_id']}")
            return existing

        # Calculate error metrics
        spread_error = None
        if game_data.get('pred_spread') is not None:
            spread_error = game_data['pred_spread'] - game_data['actual_spread']

        total_error = None
        if game_data.get('pred_total') is not None:
            total_error = game_data['pred_total'] - game_data['actual_total']

        # Calibration bucket (0-9 for 0-10%, 10-20%, etc.)
        bucket = None
        if game_data.get('pred_home_win_prob') is not None:
            bucket = min(int(game_data['pred_home_win_prob'] * 10), 9)

        training_record = TrainingData(
            game_id=game_data['game_id'],
            season=game_data['season'],
            game_date=game_data['game_date'],
            home_team_id=game_data['home_team_id'],
            away_team_id=game_data['away_team_id'],
            features=game_data['features'],
            pred_home_win_prob=game_data.get('pred_home_win_prob'),
            pred_spread=game_data.get('pred_spread'),
            pred_total=game_data.get('pred_total'),
            pred_model_version=game_data.get('pred_model_version'),
            kenpom_home_win_prob=game_data.get('kenpom_home_win_prob'),
            kenpom_spread=game_data.get('kenpom_spread'),
            kenpom_total=game_data.get('kenpom_total'),
            actual_home_score=game_data['actual_home_score'],
            actual_away_score=game_data['actual_away_score'],
            actual_spread=game_data['actual_spread'],
            actual_total=game_data['actual_total'],
            home_won=game_data['home_won'],
            spread_error=spread_error,
            total_error=total_error,
            prob_calibration_bucket=bucket,
        )

        self.session.add(training_record)
        self.session.commit()

        logger.debug(f"Stored training data for game {game_data['game_id']}")
        return training_record

    def store_prediction_audit(
        self,
        prediction_id: int,
        model_version: str,
        home_win_prob: float,
        spread_pred: float,
        total_pred: float,
        confidence: float,
        features: Dict[str, float]
    ) -> Any:
        """
        Store a prediction audit record for tracking.

        Args:
            prediction_id: ID of the Prediction record
            model_version: Version string of the model used
            home_win_prob: Predicted win probability
            spread_pred: Predicted spread
            total_pred: Predicted total
            confidence: Model confidence
            features: Features used for prediction

        Returns:
            PredictionAudit model instance
        """
        from models.database import PredictionAudit

        # Create features hash for deduplication
        features_json = json.dumps(features, sort_keys=True)
        features_hash = hashlib.sha256(features_json.encode()).hexdigest()

        audit_record = PredictionAudit(
            prediction_id=prediction_id,
            model_version=model_version,
            home_win_prob=home_win_prob,
            spread_pred=spread_pred,
            total_pred=total_pred,
            confidence=confidence,
            features_hash=features_hash,
        )

        self.session.add(audit_record)
        self.session.commit()

        return audit_record

    def get_training_dataframe(
        self,
        min_date: datetime = None,
        max_date: datetime = None,
        season: int = None
    ):
        """
        Export training data as a pandas DataFrame for model training.

        Args:
            min_date: Minimum game date filter
            max_date: Maximum game date filter
            season: Season filter

        Returns:
            pandas DataFrame with features and targets
        """
        import pandas as pd
        from models.database import TrainingData

        query = self.session.query(TrainingData)

        if min_date:
            query = query.filter(TrainingData.game_date >= min_date)
        if max_date:
            query = query.filter(TrainingData.game_date <= max_date)
        if season:
            query = query.filter(TrainingData.season == season)

        records = query.all()

        if not records:
            logger.warning("No training data found")
            return pd.DataFrame()

        # Build DataFrame
        rows = []
        for record in records:
            row = {
                'game_id': record.game_id,
                'season': record.season,
                'game_date': record.game_date,
                'home_won': record.home_won,
                'actual_spread': record.actual_spread,
                'actual_total': record.actual_total,
            }

            # Add features
            if record.features:
                row.update(record.features)

            rows.append(row)

        df = pd.DataFrame(rows)
        logger.info(f"Exported {len(df)} training records")
        return df

    def get_accuracy_stats(
        self,
        days_back: int = 30,
        model_version: str = None
    ) -> Dict[str, Any]:
        """
        Calculate prediction accuracy statistics.

        Args:
            days_back: Number of days to analyze
            model_version: Optional filter by model version

        Returns:
            Dict with accuracy metrics
        """
        from models.database import TrainingData
        import numpy as np

        cutoff = datetime.utcnow() - timedelta(days=days_back)

        query = self.session.query(TrainingData).filter(
            TrainingData.game_date >= cutoff
        )

        if model_version:
            query = query.filter(TrainingData.pred_model_version == model_version)

        records = query.all()

        if not records:
            return {
                'n_games': 0,
                'message': 'No data available'
            }

        # Calculate metrics
        n_games = len(records)

        # Win prediction accuracy (for games with predictions)
        with_preds = [r for r in records if r.pred_home_win_prob is not None]
        if with_preds:
            correct = sum(
                1 for r in with_preds
                if (r.pred_home_win_prob > 0.5) == r.home_won
            )
            win_accuracy = correct / len(with_preds)

            # Brier score
            brier = np.mean([
                (r.pred_home_win_prob - (1.0 if r.home_won else 0.0)) ** 2
                for r in with_preds
            ])

            # Spread MAE
            spread_errors = [
                abs(r.spread_error) for r in with_preds
                if r.spread_error is not None
            ]
            spread_mae = np.mean(spread_errors) if spread_errors else None

            # Total MAE
            total_errors = [
                abs(r.total_error) for r in with_preds
                if r.total_error is not None
            ]
            total_mae = np.mean(total_errors) if total_errors else None
        else:
            win_accuracy = None
            brier = None
            spread_mae = None
            total_mae = None

        # Calibration by bucket
        calibration = {}
        for bucket in range(10):
            bucket_records = [
                r for r in with_preds
                if r.prob_calibration_bucket == bucket
            ]
            if bucket_records:
                actual_rate = sum(1 for r in bucket_records if r.home_won) / len(bucket_records)
                calibration[f'{bucket*10}-{(bucket+1)*10}%'] = {
                    'predicted_range': f'{bucket*10}-{(bucket+1)*10}%',
                    'actual_win_rate': actual_rate,
                    'n_games': len(bucket_records)
                }

        return {
            'n_games': n_games,
            'n_with_predictions': len(with_preds),
            'win_accuracy': win_accuracy,
            'brier_score': brier,
            'spread_mae': spread_mae,
            'total_mae': total_mae,
            'calibration': calibration,
            'period': f'Last {days_back} days',
        }

    def run_collection(
        self,
        days_back: int = 7,
        store: bool = True
    ) -> Dict[str, Any]:
        """
        Run the full feedback collection process.

        Args:
            days_back: Days to look back
            store: Whether to store training data

        Returns:
            Summary of collection results
        """
        logger.info(f"Starting feedback collection for last {days_back} days")

        # Collect completed games
        game_data = self.collect_completed_games(days_back=days_back)

        stored_count = 0
        if store:
            for data in game_data:
                try:
                    self.store_training_data(data)
                    stored_count += 1
                except Exception as e:
                    logger.warning(f"Failed to store game {data['game_id']}: {e}")

        return {
            'games_found': len(game_data),
            'games_stored': stored_count,
            'period': f'Last {days_back} days',
        }
