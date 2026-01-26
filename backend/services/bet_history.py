"""
Bet History Service

Automatically stores value bets and tracks results after games complete.
Uses PostgreSQL for persistent storage (works with Vercel Postgres).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from models.connection import SessionLocal
from models.database import StoredBet as StoredBetModel

logger = logging.getLogger(__name__)


def _normalize_team_name(name: str) -> str:
    """Normalize team name for comparison (handles different formats)."""
    if not name:
        return ""
    name = name.lower().strip()
    # Normalize common variations
    name = name.replace(" st.", " state").replace(" st ", " state ")
    name = name.replace("'", "").replace("'", "")
    name = name.replace("-", " ")
    # Remove common suffixes
    for suffix in [" university", " college"]:
        name = name.replace(suffix, "")
    return name.strip()


def _teams_match(name1: str, name2: str) -> bool:
    """Check if two team names refer to the same team."""
    if not name1 or not name2:
        return False
    n1 = _normalize_team_name(name1)
    n2 = _normalize_team_name(name2)
    # Exact match after normalization
    if n1 == n2:
        return True
    # One contains the other (handles "North Carolina" vs "north carolina tar heels")
    if n1 in n2 or n2 in n1:
        return True
    # Check if primary words match (handles abbreviated names)
    words1 = set(w for w in n1.split() if len(w) > 3)
    words2 = set(w for w in n2.split() if len(w) > 3)
    if words1 and words2 and words1 & words2:
        return True
    return False


@dataclass
class DailyBetSummary:
    """Summary of bets for a single day."""
    date: str
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pushes: int = 0
    pending: int = 0

    # By confidence
    high_conf_bets: int = 0
    high_conf_wins: int = 0
    medium_conf_bets: int = 0
    medium_conf_wins: int = 0
    low_conf_bets: int = 0
    low_conf_wins: int = 0

    # By type
    spread_bets: int = 0
    spread_wins: int = 0
    total_bets_count: int = 0
    total_wins: int = 0
    ml_bets: int = 0
    ml_wins: int = 0

    # Metrics
    avg_edge: float = 0.0
    avg_ev: float = 0.0

    @property
    def win_rate(self) -> float:
        decided = self.wins + self.losses
        return self.wins / decided if decided > 0 else 0.0

    @property
    def high_conf_win_rate(self) -> float:
        return self.high_conf_wins / self.high_conf_bets if self.high_conf_bets > 0 else 0.0


@dataclass
class OverallStats:
    """Overall betting statistics."""
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_pushes: int = 0
    total_pending: int = 0

    win_rate: float = 0.0

    # By confidence tier
    high_conf_record: str = "0-0-0"
    high_conf_win_rate: float = 0.0
    medium_conf_record: str = "0-0-0"
    medium_conf_win_rate: float = 0.0
    low_conf_record: str = "0-0-0"
    low_conf_win_rate: float = 0.0

    # By bet type
    spread_record: str = "0-0-0"
    spread_win_rate: float = 0.0
    total_record: str = "0-0-0"
    total_win_rate: float = 0.0
    ml_record: str = "0-0-0"
    ml_win_rate: float = 0.0

    # Averages
    avg_edge: float = 0.0
    avg_ev: float = 0.0

    # Streaks
    current_streak: int = 0
    streak_type: str = ""
    longest_win_streak: int = 0
    longest_loss_streak: int = 0

    # Recent performance
    last_7_days_record: str = "0-0-0"
    last_7_days_win_rate: float = 0.0
    last_30_days_record: str = "0-0-0"
    last_30_days_win_rate: float = 0.0


class BetHistoryService:
    """
    Service for storing and tracking bet history with automatic result detection.
    Uses PostgreSQL for persistent storage.
    """

    def __init__(self):
        """Initialize the bet history service."""
        pass

    def _get_session(self) -> Session:
        """Get a database session."""
        return SessionLocal()

    def store_value_bet(
        self,
        game_data: Dict[str, Any],
        bet_data: Dict[str, Any]
    ) -> Optional[StoredBetModel]:
        """
        Store a new value bet for tracking.

        Args:
            game_data: Game information (teams, time, predictions)
            bet_data: Bet details (type, side, odds, edge, etc.)

        Returns:
            The stored bet record or None if already exists
        """
        session = self._get_session()
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Generate unique ID
            bet_id = f"{game_data['home_team']}_{game_data['away_team']}_{bet_data['type']}_{bet_data['side']}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

            # Check if this exact bet already exists for today
            existing = session.query(StoredBetModel).filter(
                StoredBetModel.home_team == game_data['home_team'],
                StoredBetModel.away_team == game_data['away_team'],
                StoredBetModel.bet_type == bet_data['type'],
                StoredBetModel.side == bet_data['side'],
                StoredBetModel.date == today
            ).first()

            if existing:
                return existing

            stored_bet = StoredBetModel(
                bet_id=bet_id,
                date=today,
                game_time=game_data.get('game_time', ''),
                home_team=game_data['home_team'],
                away_team=game_data['away_team'],
                home_rank=game_data.get('home_rank', 0),
                away_rank=game_data.get('away_rank', 0),
                bet_type=bet_data['type'],
                side=bet_data['side'],
                line=bet_data.get('line'),
                odds=bet_data.get('odds', -110),
                book=bet_data.get('book', ''),
                model_prob=bet_data.get('model_prob', 0.0),
                market_implied_prob=bet_data.get('market_implied_prob', 0.0),
                edge=bet_data.get('edge', 0.0),
                ev=bet_data.get('ev', 0.0),
                kelly=bet_data.get('kelly', 0.0),
                confidence=bet_data.get('confidence', 'low'),
                confidence_score=bet_data.get('confidence_score'),
                kenpom_spread=game_data.get('kenpom_spread', 0.0),
                kenpom_total=game_data.get('kenpom_total', 0.0),
                vegas_spread=game_data.get('vegas_spread', 0.0),
                vegas_total=game_data.get('vegas_total', 0.0),
                created_at=datetime.now(timezone.utc)
            )

            session.add(stored_bet)
            session.commit()
            session.refresh(stored_bet)
            return stored_bet

        except IntegrityError:
            session.rollback()
            logger.warning(f"Duplicate bet detected, skipping")
            return None
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing bet: {e}")
            raise
        finally:
            session.close()

    def store_all_value_bets(self, value_bets_data: List[Dict[str, Any]]) -> int:
        """
        Store all value bets from today's analysis.

        Args:
            value_bets_data: List of value bet data from the API

        Returns:
            Number of new bets stored
        """
        count = 0
        for vb in value_bets_data:
            game_data = {
                'home_team': vb['home_team'],
                'away_team': vb['away_team'],
                'game_time': vb['game_time'],
                'home_rank': vb.get('home_rank', 0),
                'away_rank': vb.get('away_rank', 0),
                'kenpom_spread': vb.get('kenpom_spread', 0),
                'kenpom_total': vb.get('kenpom_total', 0),
                'vegas_spread': vb.get('vegas_spread', 0),
                'vegas_total': vb.get('vegas_total', 0)
            }

            bet_data = {
                'type': vb['bet']['type'],
                'side': vb['bet']['side'],
                'line': vb['bet'].get('line'),
                'odds': vb['bet'].get('odds', -110),
                'book': vb['bet'].get('book', ''),
                'model_prob': vb['bet'].get('model_prob', 0.0),
                'market_implied_prob': vb['bet'].get('market_implied_prob', 0.0),
                'edge': vb['bet'].get('edge', 0.0),
                'ev': vb['bet'].get('ev', 0.0),
                'kelly': vb['bet'].get('kelly', 0.0),
                'confidence': vb['bet'].get('confidence', 'low'),
                'confidence_score': vb['bet'].get('confidence_score')
            }

            result = self.store_value_bet(game_data, bet_data)
            if result and not hasattr(result, '_existing'):
                count += 1

        return count

    def update_results(self, scores: Dict[str, Dict[str, int]]) -> int:
        """
        Update bet results based on final scores.

        Args:
            scores: Dict mapping "home_team vs away_team" to {"home": score, "away": score}

        Returns:
            Number of bets updated
        """
        session = self._get_session()
        try:
            # Get all pending bets
            pending_bets = session.query(StoredBetModel).filter(
                StoredBetModel.result.is_(None)
            ).all()

            updated = 0
            for bet in pending_bets:
                # Look up the game score
                game_key = f"{bet.home_team} vs {bet.away_team}"
                if game_key not in scores:
                    continue

                score = scores[game_key]
                home_score = score['home']
                away_score = score['away']

                bet.home_score = home_score
                bet.away_score = away_score
                bet.actual_margin = home_score - away_score
                bet.actual_total = home_score + away_score

                # Determine result
                bet.result = self._determine_result(bet)
                bet.settled_at = datetime.now(timezone.utc)
                updated += 1

            session.commit()
            return updated

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating results: {e}")
            raise
        finally:
            session.close()

    def update_single_bet_result(self, bet_id: str, home_score: int, away_score: int) -> int:
        """
        Update a single bet's result with the final score.

        Args:
            bet_id: The bet's unique identifier
            home_score: Final home team score
            away_score: Final away team score

        Returns:
            1 if updated, 0 if not found or already settled
        """
        # Validate scores - must be positive integers for a completed game
        # A score of 0-0 is not valid for basketball
        if home_score <= 0 or away_score <= 0:
            logger.warning(f"Invalid scores for bet {bet_id}: {home_score}-{away_score}, skipping")
            return 0

        session = self._get_session()
        try:
            bet = session.query(StoredBetModel).filter(
                StoredBetModel.bet_id == bet_id,
                StoredBetModel.result.is_(None)
            ).first()

            if not bet:
                return 0

            bet.home_score = home_score
            bet.away_score = away_score
            bet.actual_margin = home_score - away_score
            bet.actual_total = home_score + away_score

            # Determine result
            bet.result = self._determine_result(bet)
            bet.settled_at = datetime.now(timezone.utc)

            session.commit()
            return 1

        except Exception as e:
            session.rollback()
            logger.error(f"Error updating single bet result: {e}")
            return 0
        finally:
            session.close()

    def _determine_result(self, bet: StoredBetModel) -> Optional[str]:
        """Determine if a bet won, lost, or pushed."""
        if bet.actual_margin is None or bet.actual_total is None:
            return None

        if bet.bet_type == 'spread':
            line = bet.line or 0

            # Use fuzzy matching for team names to handle different formats
            # (e.g., "North Carolina" vs "UNC" or "north carolina")
            is_home_bet = _teams_match(bet.side, bet.home_team)

            if is_home_bet:
                # Home team spread bet
                cover_margin = bet.actual_margin + line
            else:
                # Away team spread bet
                cover_margin = -bet.actual_margin - line

            if cover_margin > 0:
                return 'win'
            elif cover_margin < 0:
                return 'loss'
            else:
                return 'push'

        elif bet.bet_type == 'total':
            line = bet.line or 0

            if bet.side == 'over':
                if bet.actual_total > line:
                    return 'win'
                elif bet.actual_total < line:
                    return 'loss'
                else:
                    return 'push'
            else:  # under
                if bet.actual_total < line:
                    return 'win'
                elif bet.actual_total > line:
                    return 'loss'
                else:
                    return 'push'

        elif bet.bet_type == 'moneyline':
            # Use fuzzy matching for team names
            is_home_bet = _teams_match(bet.side, bet.home_team)

            if is_home_bet:
                if bet.actual_margin > 0:
                    return 'win'
                elif bet.actual_margin < 0:
                    return 'loss'
                else:
                    return 'push'
            else:
                if bet.actual_margin < 0:
                    return 'win'
                elif bet.actual_margin > 0:
                    return 'loss'
                else:
                    return 'push'

        return None

    def get_bets_by_date(self, date: str) -> List[StoredBetModel]:
        """Get all bets for a specific date."""
        session = self._get_session()
        try:
            return session.query(StoredBetModel).filter(
                StoredBetModel.date == date
            ).all()
        finally:
            session.close()

    def get_pending_bets(self) -> List[StoredBetModel]:
        """Get all bets awaiting results."""
        session = self._get_session()
        try:
            return session.query(StoredBetModel).filter(
                StoredBetModel.result.is_(None)
            ).all()
        finally:
            session.close()

    def get_recent_bets(self, days: int = 7) -> List[StoredBetModel]:
        """Get bets from the last N days."""
        session = self._get_session()
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            return session.query(StoredBetModel).filter(
                StoredBetModel.date >= cutoff
            ).order_by(StoredBetModel.created_at.desc()).all()
        finally:
            session.close()

    def get_all_bets(self) -> List[StoredBetModel]:
        """Get all bets."""
        session = self._get_session()
        try:
            return session.query(StoredBetModel).order_by(
                StoredBetModel.created_at.desc()
            ).all()
        finally:
            session.close()

    def get_daily_summary(self, date: str) -> DailyBetSummary:
        """Get summary statistics for a specific day."""
        day_bets = self.get_bets_by_date(date)

        summary = DailyBetSummary(date=date)
        summary.total_bets = len(day_bets)

        edges = []
        evs = []

        for bet in day_bets:
            edges.append(bet.edge or 0)
            evs.append(bet.ev or 0)

            if bet.result == 'win':
                summary.wins += 1
            elif bet.result == 'loss':
                summary.losses += 1
            elif bet.result == 'push':
                summary.pushes += 1
            else:
                summary.pending += 1

            # By confidence
            if bet.confidence == 'high':
                summary.high_conf_bets += 1
                if bet.result == 'win':
                    summary.high_conf_wins += 1
            elif bet.confidence == 'medium':
                summary.medium_conf_bets += 1
                if bet.result == 'win':
                    summary.medium_conf_wins += 1
            else:
                summary.low_conf_bets += 1
                if bet.result == 'win':
                    summary.low_conf_wins += 1

            # By type
            if bet.bet_type == 'spread':
                summary.spread_bets += 1
                if bet.result == 'win':
                    summary.spread_wins += 1
            elif bet.bet_type == 'total':
                summary.total_bets_count += 1
                if bet.result == 'win':
                    summary.total_wins += 1
            elif bet.bet_type == 'moneyline':
                summary.ml_bets += 1
                if bet.result == 'win':
                    summary.ml_wins += 1

        summary.avg_edge = sum(edges) / len(edges) if edges else 0.0
        summary.avg_ev = sum(evs) / len(evs) if evs else 0.0

        return summary

    def get_overall_stats(self) -> OverallStats:
        """Calculate overall betting statistics."""
        bets = self.get_all_bets()

        stats = OverallStats()
        stats.total_bets = len(bets)

        # Track by category
        high_w, high_l, high_p = 0, 0, 0
        med_w, med_l, med_p = 0, 0, 0
        low_w, low_l, low_p = 0, 0, 0

        spread_w, spread_l, spread_p = 0, 0, 0
        total_w, total_l, total_p = 0, 0, 0
        ml_w, ml_l, ml_p = 0, 0, 0

        edges = []
        evs = []

        # For streaks
        results_in_order = []

        for bet in sorted(bets, key=lambda x: x.created_at or datetime.min):
            edges.append(bet.edge or 0)
            evs.append(bet.ev or 0)

            if bet.result == 'win':
                stats.total_wins += 1
                results_in_order.append('W')
            elif bet.result == 'loss':
                stats.total_losses += 1
                results_in_order.append('L')
            elif bet.result == 'push':
                stats.total_pushes += 1
            else:
                stats.total_pending += 1

            # By confidence
            if bet.confidence == 'high':
                if bet.result == 'win':
                    high_w += 1
                elif bet.result == 'loss':
                    high_l += 1
                elif bet.result == 'push':
                    high_p += 1
            elif bet.confidence == 'medium':
                if bet.result == 'win':
                    med_w += 1
                elif bet.result == 'loss':
                    med_l += 1
                elif bet.result == 'push':
                    med_p += 1
            else:
                if bet.result == 'win':
                    low_w += 1
                elif bet.result == 'loss':
                    low_l += 1
                elif bet.result == 'push':
                    low_p += 1

            # By type
            if bet.bet_type == 'spread':
                if bet.result == 'win':
                    spread_w += 1
                elif bet.result == 'loss':
                    spread_l += 1
                elif bet.result == 'push':
                    spread_p += 1
            elif bet.bet_type == 'total':
                if bet.result == 'win':
                    total_w += 1
                elif bet.result == 'loss':
                    total_l += 1
                elif bet.result == 'push':
                    total_p += 1
            elif bet.bet_type == 'moneyline':
                if bet.result == 'win':
                    ml_w += 1
                elif bet.result == 'loss':
                    ml_l += 1
                elif bet.result == 'push':
                    ml_p += 1

        # Calculate rates
        decided = stats.total_wins + stats.total_losses
        stats.win_rate = stats.total_wins / decided if decided > 0 else 0.0

        # Records
        stats.high_conf_record = f"{high_w}-{high_l}-{high_p}"
        stats.high_conf_win_rate = high_w / (high_w + high_l) if (high_w + high_l) > 0 else 0.0

        stats.medium_conf_record = f"{med_w}-{med_l}-{med_p}"
        stats.medium_conf_win_rate = med_w / (med_w + med_l) if (med_w + med_l) > 0 else 0.0

        stats.low_conf_record = f"{low_w}-{low_l}-{low_p}"
        stats.low_conf_win_rate = low_w / (low_w + low_l) if (low_w + low_l) > 0 else 0.0

        stats.spread_record = f"{spread_w}-{spread_l}-{spread_p}"
        stats.spread_win_rate = spread_w / (spread_w + spread_l) if (spread_w + spread_l) > 0 else 0.0

        stats.total_record = f"{total_w}-{total_l}-{total_p}"
        stats.total_win_rate = total_w / (total_w + total_l) if (total_w + total_l) > 0 else 0.0

        stats.ml_record = f"{ml_w}-{ml_l}-{ml_p}"
        stats.ml_win_rate = ml_w / (ml_w + ml_l) if (ml_w + ml_l) > 0 else 0.0

        # Averages
        stats.avg_edge = sum(edges) / len(edges) if edges else 0.0
        stats.avg_ev = sum(evs) / len(evs) if evs else 0.0

        # Streaks
        if results_in_order:
            stats.current_streak = 1
            stats.streak_type = results_in_order[-1]
            for i in range(len(results_in_order) - 2, -1, -1):
                if results_in_order[i] == stats.streak_type:
                    stats.current_streak += 1
                else:
                    break

            # Longest streaks
            current_w, current_l = 0, 0
            for r in results_in_order:
                if r == 'W':
                    current_w += 1
                    current_l = 0
                    stats.longest_win_streak = max(stats.longest_win_streak, current_w)
                else:
                    current_l += 1
                    current_w = 0
                    stats.longest_loss_streak = max(stats.longest_loss_streak, current_l)

        # Recent performance
        last_7 = self.get_recent_bets(7)
        w7 = len([b for b in last_7 if b.result == 'win'])
        l7 = len([b for b in last_7 if b.result == 'loss'])
        p7 = len([b for b in last_7 if b.result == 'push'])
        stats.last_7_days_record = f"{w7}-{l7}-{p7}"
        stats.last_7_days_win_rate = w7 / (w7 + l7) if (w7 + l7) > 0 else 0.0

        last_30 = self.get_recent_bets(30)
        w30 = len([b for b in last_30 if b.result == 'win'])
        l30 = len([b for b in last_30 if b.result == 'loss'])
        p30 = len([b for b in last_30 if b.result == 'push'])
        stats.last_30_days_record = f"{w30}-{l30}-{p30}"
        stats.last_30_days_win_rate = w30 / (w30 + l30) if (w30 + l30) > 0 else 0.0

        return stats

    def get_history_for_display(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get bet history formatted for frontend display."""
        recent_bets = self.get_recent_bets(days)

        return [
            {
                'id': bet.bet_id,
                'date': bet.date,
                'game_time': bet.game_time,
                'home_team': bet.home_team,
                'away_team': bet.away_team,
                'home_rank': bet.home_rank,
                'away_rank': bet.away_rank,
                'bet': {
                    'type': bet.bet_type,
                    'side': bet.side,
                    'line': bet.line,
                    'odds': bet.odds,
                    'book': bet.book,
                    'edge': bet.edge,
                    'ev': bet.ev,
                    'kelly': bet.kelly,
                    'confidence': bet.confidence,
                    'confidence_score': bet.confidence_score
                },
                'kenpom_spread': bet.kenpom_spread,
                'kenpom_total': bet.kenpom_total,
                'vegas_spread': bet.vegas_spread,
                'vegas_total': bet.vegas_total,
                'result': bet.result,
                'home_score': bet.home_score,
                'away_score': bet.away_score,
                'actual_margin': bet.actual_margin,
                'actual_total': bet.actual_total
            }
            for bet in recent_bets
        ]

    def clear_old_bets(self, days: int = 90) -> int:
        """Remove bets older than specified days."""
        session = self._get_session()
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
            result = session.query(StoredBetModel).filter(
                StoredBetModel.date < cutoff
            ).delete()
            session.commit()
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing old bets: {e}")
            raise
        finally:
            session.close()
