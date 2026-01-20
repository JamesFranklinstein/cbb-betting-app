"""
Database Models

SQLAlchemy models for storing games, predictions, odds, and tracking results.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Text, JSON, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()


class Team(Base):
    """College basketball team."""
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True)
    kenpom_id = Column(Integer, unique=True, nullable=True)
    name = Column(String(255), nullable=False)
    conference = Column(String(50))
    coach = Column(String(255))
    arena = Column(String(255))
    
    # Relationships
    home_games = relationship("Game", back_populates="home_team", foreign_keys="Game.home_team_id")
    away_games = relationship("Game", back_populates="away_team", foreign_keys="Game.away_team_id")
    ratings = relationship("TeamRating", back_populates="team")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TeamRating(Base):
    """Historical KenPom ratings for a team."""
    __tablename__ = "team_ratings"
    
    id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    season = Column(Integer, nullable=False)
    
    # KenPom metrics
    adj_em = Column(Float)          # Adjusted Efficiency Margin
    rank_adj_em = Column(Integer)
    adj_oe = Column(Float)          # Adjusted Offensive Efficiency
    rank_adj_oe = Column(Integer)
    adj_de = Column(Float)          # Adjusted Defensive Efficiency
    rank_adj_de = Column(Integer)
    adj_tempo = Column(Float)       # Adjusted Tempo
    rank_adj_tempo = Column(Integer)
    
    # Additional metrics
    luck = Column(Float)
    sos = Column(Float)             # Strength of Schedule
    pythag = Column(Float)          # Pythagorean expectation
    
    # Four Factors (Offense)
    efg_pct = Column(Float)         # Effective FG%
    to_pct = Column(Float)          # Turnover %
    or_pct = Column(Float)          # Offensive Rebound %
    ft_rate = Column(Float)         # Free Throw Rate
    
    # Four Factors (Defense)
    d_efg_pct = Column(Float)
    d_to_pct = Column(Float)
    d_or_pct = Column(Float)
    d_ft_rate = Column(Float)
    
    team = relationship("Team", back_populates="ratings")
    
    __table_args__ = (
        UniqueConstraint("team_id", "date", name="unique_team_date_rating"),
        Index("idx_team_rating_date", "team_id", "date"),
    )


class Game(Base):
    """A college basketball game."""
    __tablename__ = "games"
    
    id = Column(Integer, primary_key=True)
    external_id = Column(String(100), unique=True)  # Odds API event ID
    kenpom_game_id = Column(Integer, nullable=True)
    
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    
    scheduled_time = Column(DateTime, nullable=False)
    season = Column(Integer, nullable=False)
    
    # Actual results (filled in after game)
    home_score = Column(Integer)
    away_score = Column(Integer)
    is_completed = Column(Boolean, default=False)
    
    # Relationships
    home_team = relationship("Team", back_populates="home_games", foreign_keys=[home_team_id])
    away_team = relationship("Team", back_populates="away_games", foreign_keys=[away_team_id])
    predictions = relationship("Prediction", back_populates="game")
    odds_snapshots = relationship("OddsSnapshot", back_populates="game")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_game_scheduled", "scheduled_time"),
        Index("idx_game_season", "season"),
    )


class Prediction(Base):
    """A prediction for a game (KenPom or ML model)."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    
    source = Column(String(50), nullable=False)  # 'kenpom', 'ml_v1', etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Score predictions
    home_score_pred = Column(Float)
    away_score_pred = Column(Float)
    
    # Win probability
    home_win_prob = Column(Float)
    
    # Spread prediction (positive = home favored)
    spread_pred = Column(Float)
    
    # Total prediction
    total_pred = Column(Float)
    
    # Tempo prediction
    tempo_pred = Column(Float)
    
    # Model confidence (for ML predictions)
    confidence = Column(Float)
    
    # Feature data used for prediction (for ML model debugging)
    features = Column(JSON)
    
    game = relationship("Game", back_populates="predictions")
    value_bets = relationship("ValueBet", back_populates="prediction")
    
    __table_args__ = (
        UniqueConstraint("game_id", "source", "created_at", name="unique_game_source_time"),
    )


class OddsSnapshot(Base):
    """Snapshot of betting odds at a point in time."""
    __tablename__ = "odds_snapshots"
    
    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    captured_at = Column(DateTime, default=datetime.utcnow)
    
    # Consensus/average lines
    spread_home = Column(Float)      # e.g., -6.5
    spread_home_odds = Column(Integer)  # e.g., -110
    spread_away_odds = Column(Integer)
    
    total = Column(Float)            # e.g., 145.5
    over_odds = Column(Integer)
    under_odds = Column(Integer)
    
    moneyline_home = Column(Integer)  # e.g., -250
    moneyline_away = Column(Integer)  # e.g., +200
    
    # Best available odds across books
    best_spread_home_odds = Column(Integer)
    best_spread_home_book = Column(String(50))
    best_spread_away_odds = Column(Integer)
    best_spread_away_book = Column(String(50))
    
    best_over_odds = Column(Integer)
    best_over_book = Column(String(50))
    best_under_odds = Column(Integer)
    best_under_book = Column(String(50))
    
    best_ml_home_odds = Column(Integer)
    best_ml_home_book = Column(String(50))
    best_ml_away_odds = Column(Integer)
    best_ml_away_book = Column(String(50))
    
    # Raw data from API
    raw_data = Column(JSON)
    
    game = relationship("Game", back_populates="odds_snapshots")
    
    __table_args__ = (
        Index("idx_odds_game_time", "game_id", "captured_at"),
    )


class ValueBet(Base):
    """Identified value betting opportunity."""
    __tablename__ = "value_bets"
    
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    
    bet_type = Column(String(20), nullable=False)  # 'spread', 'total', 'moneyline'
    side = Column(String(50), nullable=False)       # team name, 'over', 'under'
    
    # Our prediction
    model_prob = Column(Float, nullable=False)      # Our predicted probability
    model_line = Column(Float)                       # Our predicted line
    
    # Market odds
    market_odds = Column(Integer, nullable=False)
    market_line = Column(Float)
    market_implied_prob = Column(Float, nullable=False)
    
    # Value calculation
    edge = Column(Float, nullable=False)            # model_prob - market_implied_prob
    kelly_fraction = Column(Float)                   # Kelly criterion bet size
    
    # Best book to place bet
    recommended_book = Column(String(50))
    
    # Result tracking
    is_winner = Column(Boolean)
    actual_result = Column(Float)  # Actual margin or total
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    prediction = relationship("Prediction", back_populates="value_bets")
    game = relationship("Game")
    
    __table_args__ = (
        Index("idx_value_bet_type", "bet_type"),
        Index("idx_value_bet_edge", "edge"),
    )


class BettingResult(Base):
    """Track actual betting performance."""
    __tablename__ = "betting_results"
    
    id = Column(Integer, primary_key=True)
    value_bet_id = Column(Integer, ForeignKey("value_bets.id"), nullable=False)
    
    # Bet details
    stake = Column(Float)           # Amount wagered (in units)
    odds_taken = Column(Integer)    # Actual odds when bet placed
    
    # Result
    payout = Column(Float)          # Amount won/lost (in units)
    is_win = Column(Boolean)
    
    # Running totals (updated after each bet)
    cumulative_profit = Column(Float)
    cumulative_roi = Column(Float)
    
    settled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_betting_result_settled", "settled_at"),
    )


class MLModelVersion(Base):
    """Track ML model versions and performance."""
    __tablename__ = "ml_model_versions"

    id = Column(Integer, primary_key=True)
    version = Column(String(50), unique=True, nullable=False)

    # Model details
    model_type = Column(String(100))  # e.g., 'XGBoost', 'RandomForest', 'PyTorch'
    features_used = Column(JSON)
    hyperparameters = Column(JSON)

    # Training metrics
    train_accuracy = Column(Float)
    val_accuracy = Column(Float)
    train_log_loss = Column(Float)
    val_log_loss = Column(Float)

    # Calibration metrics
    brier_score = Column(Float)
    calibration_error = Column(Float)

    # Spread/Total regression metrics
    spread_mae = Column(Float)
    total_mae = Column(Float)

    # Live performance
    live_accuracy = Column(Float)
    live_roi = Column(Float)
    total_predictions = Column(Integer, default=0)

    is_active = Column(Boolean, default=False)
    trained_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Path to saved model file
    model_path = Column(String(500))


class TrainingData(Base):
    """Historical game data with predictions and outcomes for model training."""
    __tablename__ = "training_data"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    season = Column(Integer, nullable=False)
    game_date = Column(DateTime, nullable=False)

    # Teams
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    # Features snapshot (JSON blob of all features at prediction time)
    features = Column(JSON, nullable=False)

    # Our ML predictions at game time
    pred_home_win_prob = Column(Float)
    pred_spread = Column(Float)
    pred_total = Column(Float)
    pred_model_version = Column(String(50))

    # KenPom predictions for comparison
    kenpom_home_win_prob = Column(Float)
    kenpom_spread = Column(Float)
    kenpom_total = Column(Float)

    # Actual outcomes
    actual_home_score = Column(Integer)
    actual_away_score = Column(Integer)
    actual_spread = Column(Float)  # home_score - away_score
    actual_total = Column(Float)   # home_score + away_score
    home_won = Column(Boolean)

    # Error metrics (for analysis)
    spread_error = Column(Float)   # pred_spread - actual_spread
    total_error = Column(Float)    # pred_total - actual_total
    prob_calibration_bucket = Column(Integer)  # 0-9 for 0-10%, 10-20%, etc.

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    game = relationship("Game")
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

    __table_args__ = (
        Index("idx_training_data_season", "season"),
        Index("idx_training_data_date", "game_date"),
        UniqueConstraint("game_id", name="unique_training_data_game"),
    )


class PredictionAudit(Base):
    """Audit trail of all predictions for tracking accuracy over time."""
    __tablename__ = "prediction_audit"

    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    model_version = Column(String(50), nullable=False)

    # Prediction details
    home_win_prob = Column(Float)
    spread_pred = Column(Float)
    total_pred = Column(Float)
    confidence = Column(Float)

    # Features used (for debugging)
    features_hash = Column(String(64))  # SHA256 of features JSON

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    prediction = relationship("Prediction")

    __table_args__ = (
        Index("idx_prediction_audit_version", "model_version"),
        Index("idx_prediction_audit_created", "created_at"),
    )


class StoredBet(Base):
    """A value bet that was identified and stored for tracking results."""
    __tablename__ = "stored_bets"

    id = Column(Integer, primary_key=True)
    bet_id = Column(String(255), unique=True, nullable=False)  # Unique identifier
    date = Column(String(10), nullable=False)                   # YYYY-MM-DD
    game_time = Column(String(50))                              # ISO format

    # Teams
    home_team = Column(String(255), nullable=False)
    away_team = Column(String(255), nullable=False)
    home_rank = Column(Integer, default=0)
    away_rank = Column(Integer, default=0)

    # Bet details
    bet_type = Column(String(20), nullable=False)              # spread, total, moneyline
    side = Column(String(100), nullable=False)                 # team name, over, under
    line = Column(Float)                                        # The line (spread or total)
    odds = Column(Integer, default=-110)                        # American odds
    book = Column(String(100))                                  # Recommended book

    # Model predictions
    model_prob = Column(Float, default=0.0)
    market_implied_prob = Column(Float, default=0.0)
    edge = Column(Float, default=0.0)
    ev = Column(Float, default=0.0)
    kelly = Column(Float, default=0.0)
    confidence = Column(String(20), default='low')
    confidence_score = Column(Float)

    # KenPom vs Vegas
    kenpom_spread = Column(Float, default=0.0)
    kenpom_total = Column(Float, default=0.0)
    vegas_spread = Column(Float, default=0.0)
    vegas_total = Column(Float, default=0.0)

    # Result (filled in after game completes)
    result = Column(String(10))                                 # win, loss, push, None
    home_score = Column(Integer)
    away_score = Column(Integer)
    actual_margin = Column(Float)
    actual_total = Column(Float)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    settled_at = Column(DateTime)

    __table_args__ = (
        Index("idx_stored_bet_date", "date"),
        Index("idx_stored_bet_result", "result"),
        Index("idx_stored_bet_confidence", "confidence"),
        UniqueConstraint("home_team", "away_team", "bet_type", "side", "date",
                        name="unique_bet_per_game_day"),
    )
