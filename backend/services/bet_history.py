"""
Bet History Service

Automatically stores value bets and tracks results after games complete.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class StoredBet:
    """A value bet that was identified and stored for tracking."""
    id: str
    date: str                          # Date bet was identified (YYYY-MM-DD)
    game_time: str                     # ISO format game time
    home_team: str
    away_team: str
    home_rank: int
    away_rank: int

    # Bet details
    bet_type: str                      # spread, total, moneyline
    side: str                          # team name, over, under
    line: Optional[float]              # The line (spread or total)
    odds: int                          # American odds
    book: str                          # Recommended book

    # Model predictions
    model_prob: float
    market_implied_prob: float
    edge: float
    ev: float
    kelly: float
    confidence: str
    confidence_score: Optional[float] = None

    # KenPom vs Vegas
    kenpom_spread: float = 0.0
    kenpom_total: float = 0.0
    vegas_spread: float = 0.0
    vegas_total: float = 0.0

    # Result (filled in after game completes)
    result: Optional[str] = None       # win, loss, push
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    actual_margin: Optional[float] = None
    actual_total: Optional[float] = None

    # Timestamps
    created_at: str = ""
    settled_at: Optional[str] = None


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
    total_bets_count: int = 0  # Renamed to avoid conflict
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
        decided = self.high_conf_bets - self.pushes  # Approximate
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
    streak_type: str = ""  # "W" or "L"
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
    """

    def __init__(self, storage_path: str = None):
        """Initialize the bet history service."""
        if storage_path is None:
            home_dir = os.path.expanduser("~")
            storage_dir = os.path.join(home_dir, ".cbb_betting")
            os.makedirs(storage_dir, exist_ok=True)
            storage_path = os.path.join(storage_dir, "bet_history.json")

        self.storage_path = storage_path
        self.bets: List[StoredBet] = []
        self._load_bets()

    def _load_bets(self) -> None:
        """Load bets from storage file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.bets = [StoredBet(**bet) for bet in data]
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Could not load bet history: {e}")
                self.bets = []

    def _save_bets(self) -> None:
        """Save bets to storage file."""
        data = [asdict(bet) for bet in self.bets]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def store_value_bet(
        self,
        game_data: Dict[str, Any],
        bet_data: Dict[str, Any]
    ) -> StoredBet:
        """
        Store a new value bet for tracking.

        Args:
            game_data: Game information (teams, time, predictions)
            bet_data: Bet details (type, side, odds, edge, etc.)

        Returns:
            The stored bet record
        """
        # Generate unique ID
        bet_id = f"{game_data['home_team']}_{game_data['away_team']}_{bet_data['type']}_{bet_data['side']}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Check if this exact bet already exists for today
        today = datetime.now().strftime("%Y-%m-%d")
        existing = self._find_existing_bet(
            game_data['home_team'],
            game_data['away_team'],
            bet_data['type'],
            bet_data['side'],
            today
        )

        if existing:
            # Update existing bet instead of creating duplicate
            return existing

        stored_bet = StoredBet(
            id=bet_id,
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
            created_at=datetime.now().isoformat()
        )

        self.bets.append(stored_bet)
        self._save_bets()
        return stored_bet

    def _find_existing_bet(
        self,
        home_team: str,
        away_team: str,
        bet_type: str,
        side: str,
        date: str
    ) -> Optional[StoredBet]:
        """Find an existing bet matching these criteria."""
        for bet in self.bets:
            if (bet.home_team == home_team and
                bet.away_team == away_team and
                bet.bet_type == bet_type and
                bet.side == side and
                bet.date == date):
                return bet
        return None

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

            existing = self._find_existing_bet(
                game_data['home_team'],
                game_data['away_team'],
                bet_data['type'],
                bet_data['side'],
                datetime.now().strftime("%Y-%m-%d")
            )

            if not existing:
                self.store_value_bet(game_data, bet_data)
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
        updated = 0

        for bet in self.bets:
            if bet.result is not None:
                continue  # Already settled

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
            bet.settled_at = datetime.now().isoformat()
            updated += 1

        if updated > 0:
            self._save_bets()

        return updated

    def _determine_result(self, bet: StoredBet) -> str:
        """Determine if a bet won, lost, or pushed."""
        if bet.actual_margin is None or bet.actual_total is None:
            return None

        if bet.bet_type == 'spread':
            # Spread bet: did the side cover?
            line = bet.line or 0

            if bet.side == bet.home_team:
                # Home team spread bet
                # Home covers if actual_margin > -line
                # E.g., line = -6.5 (home -6.5), home covers if margin > 6.5
                cover_margin = bet.actual_margin + line
            else:
                # Away team spread bet
                # Away covers if -actual_margin > line (or actual_margin < -line)
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
            if bet.side == bet.home_team:
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

    def get_bets_by_date(self, date: str) -> List[StoredBet]:
        """Get all bets for a specific date."""
        return [bet for bet in self.bets if bet.date == date]

    def get_pending_bets(self) -> List[StoredBet]:
        """Get all bets awaiting results."""
        return [bet for bet in self.bets if bet.result is None]

    def get_recent_bets(self, days: int = 7) -> List[StoredBet]:
        """Get bets from the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        return [bet for bet in self.bets if bet.date >= cutoff]

    def get_daily_summary(self, date: str) -> DailyBetSummary:
        """Get summary statistics for a specific day."""
        day_bets = self.get_bets_by_date(date)

        summary = DailyBetSummary(date=date)
        summary.total_bets = len(day_bets)

        edges = []
        evs = []

        for bet in day_bets:
            edges.append(bet.edge)
            evs.append(bet.ev)

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
        stats = OverallStats()
        stats.total_bets = len(self.bets)

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

        for bet in sorted(self.bets, key=lambda x: x.created_at):
            edges.append(bet.edge)
            evs.append(bet.ev)

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

        # Sort by date descending, then by game time
        recent_bets.sort(key=lambda x: (x.date, x.game_time), reverse=True)

        return [
            {
                'id': bet.id,
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
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        original_count = len(self.bets)
        self.bets = [bet for bet in self.bets if bet.date >= cutoff]
        removed = original_count - len(self.bets)
        if removed > 0:
            self._save_bets()
        return removed
