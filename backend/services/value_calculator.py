"""
Value Bet Calculator

Core logic for identifying value betting opportunities by comparing
KenPom predictions (and ML model predictions) to Vegas betting lines.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import math
import json
import os
from datetime import datetime, timezone


class BetType(Enum):
    SPREAD = "spread"
    TOTAL = "total"
    MONEYLINE = "moneyline"


@dataclass
class TeamStats:
    """KenPom statistics for a single team."""
    team_name: str
    rank: int = 0

    # Efficiency ratings
    adj_em: float = 0.0      # Adjusted Efficiency Margin
    adj_oe: float = 0.0      # Adjusted Offensive Efficiency
    adj_de: float = 0.0      # Adjusted Defensive Efficiency
    adj_tempo: float = 0.0   # Adjusted Tempo

    # Four Factors - Offense
    efg_pct: float = 0.0     # Effective Field Goal %
    to_pct: float = 0.0      # Turnover %
    or_pct: float = 0.0      # Offensive Rebound %
    ft_rate: float = 0.0     # Free Throw Rate

    # Four Factors - Defense (opponent's numbers)
    d_efg_pct: float = 0.0   # Opponent eFG%
    d_to_pct: float = 0.0    # Opponent TO% (forced turnovers)
    d_or_pct: float = 0.0    # Opponent OR% (defensive rebounding)
    d_ft_rate: float = 0.0   # Opponent FT Rate

    # Additional stats
    sos: float = 0.0         # Strength of Schedule
    luck: float = 0.0        # Luck rating

    # NEW: Pythagorean rating (better win probability predictor)
    pythag: float = 0.0      # Pythagorean win expectation (0-1)
    rank_pythag: int = 0     # National rank by Pythag

    # NEW: Average Possession Length (for totals predictions)
    apl_off: float = 0.0     # Offensive possession length (seconds)
    apl_def: float = 0.0     # Defensive possession length (seconds)
    rank_apl_off: int = 0    # Rank in offensive APL
    rank_apl_def: int = 0    # Rank in defensive APL

    # NEW: Conference-specific APL
    conf_apl_off: float = 0.0   # Conference-only offensive APL
    conf_apl_def: float = 0.0   # Conference-only defensive APL

    # NEW: Strength of Schedule Components
    sos_off: float = 0.0     # Offensive SOS (opponent's defensive efficiency)
    sos_def: float = 0.0     # Defensive SOS (opponent's offensive efficiency)
    rank_sos: int = 0        # National rank by SOS
    rank_sos_off: int = 0    # Rank in offensive SOS
    rank_sos_def: int = 0    # Rank in defensive SOS
    nc_sos: float = 0.0      # Non-conference SOS
    rank_nc_sos: int = 0     # Rank in non-conference SOS

    # NEW: Conference info
    conference: str = ""     # Conference short name (e.g., "B10", "SEC")

    # NEW: Trend data (rating change over time)
    adj_em_trend: float = 0.0    # Change in AdjEM over last 2 weeks
    adj_oe_trend: float = 0.0    # Change in AdjOE over last 2 weeks
    adj_de_trend: float = 0.0    # Change in AdjDE over last 2 weeks

    # Point Distribution - Offense (% of points from each source)
    off_ft_pct: float = 0.0  # % of points from free throws
    off_2pt_pct: float = 0.0 # % of points from 2-pointers
    off_3pt_pct: float = 0.0 # % of points from 3-pointers

    # Point Distribution - Defense (opponent's point sources)
    def_ft_pct: float = 0.0
    def_2pt_pct: float = 0.0
    def_3pt_pct: float = 0.0

    # Height & Experience
    avg_height: float = 0.0      # Average height in inches
    experience: float = 0.0      # Years of experience
    bench_minutes: float = 0.0   # % of minutes from bench
    continuity: float = 0.0      # % of minutes returning

    # Misc Stats
    fg_pct_2pt: float = 0.0      # 2-point FG%
    fg_pct_3pt: float = 0.0      # 3-point FG%
    ft_pct: float = 0.0          # Free throw %
    blk_pct: float = 0.0         # Block %
    stl_pct: float = 0.0         # Steal %
    ast_ratio: float = 0.0       # Assist ratio
    opp_fg_pct_2pt: float = 0.0  # Opponent 2-point FG%
    opp_fg_pct_3pt: float = 0.0  # Opponent 3-point FG%

    # Derived metrics for predictions
    @property
    def three_point_reliance(self) -> float:
        """How much team relies on 3-pointers (0-1 scale)."""
        return self.off_3pt_pct / 100.0 if self.off_3pt_pct > 0 else 0.0

    @property
    def interior_offense(self) -> float:
        """Strength of interior offense (combined 2pt% and rebounding)."""
        return (self.fg_pct_2pt * 0.6 + self.or_pct * 0.4) if self.fg_pct_2pt > 0 else 0.0

    @property
    def defensive_pressure(self) -> float:
        """Overall defensive pressure (steals + forced TOs)."""
        return (self.stl_pct * 0.5 + self.d_to_pct * 0.5) if self.stl_pct > 0 else 0.0

    # NEW: Derived metrics using APL and Pythag
    @property
    def expected_game_pace(self) -> float:
        """Expected pace contribution based on APL (lower APL = faster pace)."""
        if self.apl_off > 0 and self.apl_def > 0:
            # Average of offensive and defensive possession lengths
            return (self.apl_off + self.apl_def) / 2
        return 0.0

    @property
    def pace_factor(self) -> float:
        """Pace factor for totals (1.0 = average, >1 = faster, <1 = slower).

        Average D1 possession is ~17 seconds. Teams with shorter possessions
        lead to more possessions per game = higher totals.
        """
        if self.expected_game_pace > 0:
            avg_pace = 17.0  # Average D1 possession length
            return avg_pace / self.expected_game_pace
        return 1.0

    @property
    def sos_adjusted_rating(self) -> float:
        """Efficiency margin adjusted for strength of schedule.

        Teams with high SOS and good AdjEM are more reliable.
        """
        if self.adj_em != 0 and self.sos != 0:
            # Boost rating slightly for teams with tougher schedules
            sos_factor = 1 + (self.sos / 100)  # SOS typically ranges -10 to +15
            return self.adj_em * sos_factor
        return self.adj_em

    @property
    def effective_height(self) -> float:
        """Height weighted by roster continuity.

        Teams with more returning players have more stable/reliable height metrics.
        """
        if self.avg_height > 0 and self.continuity > 0:
            return self.avg_height * (self.continuity / 100)
        return self.avg_height

    @property
    def is_trending_up(self) -> bool:
        """Whether team is improving (positive AdjEM trend)."""
        return self.adj_em_trend > 0.5  # At least 0.5 point improvement

    @property
    def is_trending_down(self) -> bool:
        """Whether team is declining (negative AdjEM trend)."""
        return self.adj_em_trend < -0.5  # At least 0.5 point decline

    @property
    def schedule_strength_tier(self) -> str:
        """Categorize schedule strength for confidence adjustments."""
        if self.rank_sos <= 50:
            return "elite"
        elif self.rank_sos <= 100:
            return "strong"
        elif self.rank_sos <= 200:
            return "average"
        else:
            return "weak"

    @property
    def height_tier(self) -> str:
        """Categorize team height (D1 average is ~77 inches / 6'5").

        Used for identifying height mismatches between teams.
        """
        if self.avg_height >= 79:  # 6'7" or taller
            return "tall"
        elif self.avg_height >= 77:  # 6'5" to 6'7"
            return "average"
        elif self.avg_height >= 75:  # 6'3" to 6'5"
            return "short"
        else:
            return "very_short"


@dataclass
class StatDifference:
    """Difference between two teams for a statistic."""
    stat_name: str
    display_name: str
    home_value: float
    away_value: float
    difference: float
    advantage: str  # "home", "away", or "neutral"
    significance: str  # "major", "moderate", "minor"
    higher_is_better: bool = True  # For stats like TO%, lower is better

    @property
    def is_major(self) -> bool:
        return self.significance == "major"


@dataclass
class TeamStatComparison:
    """Comparison of all statistics between two teams."""
    home_stats: TeamStats
    away_stats: TeamStats
    stat_differences: List[StatDifference] = field(default_factory=list)

    # Summary metrics
    efficiency_edge: str = ""  # Which team has efficiency edge
    tempo_mismatch: bool = False  # Significant tempo difference
    shooting_edge: str = ""  # Which team shoots better
    rebounding_edge: str = ""  # Which team rebounds better
    turnover_edge: str = ""  # Which team is better with turnovers

    @property
    def major_differences(self) -> List[StatDifference]:
        """Return only major statistical differences."""
        return [d for d in self.stat_differences if d.is_major]


@dataclass
class GamePrediction:
    """Prediction data for a game."""
    home_team: str
    away_team: str
    home_score: float
    away_score: float
    home_win_prob: float
    tempo: float = 0.0
    source: str = "kenpom"

    # Team statistics comparison (optional)
    stat_comparison: Optional[TeamStatComparison] = None

    @property
    def spread(self) -> float:
        """Predicted spread (negative = home favored, matching betting convention)."""
        return self.away_score - self.home_score

    @property
    def total(self) -> float:
        """Predicted total points."""
        return self.home_score + self.away_score

    @property
    def away_win_prob(self) -> float:
        return 1 - self.home_win_prob


@dataclass
class MarketOdds:
    """Current betting market odds for a game."""
    home_team: str
    away_team: str
    
    # Spread
    spread: float = 0.0              # Home team spread (e.g., -6.5)
    spread_home_odds: int = -110
    spread_away_odds: int = -110
    
    # Total
    total: float = 0.0
    over_odds: int = -110
    under_odds: int = -110
    
    # Moneyline
    ml_home: int = 0
    ml_away: int = 0
    
    # Best available
    best_spread_home_odds: int = -110
    best_spread_home_book: str = ""
    best_spread_away_odds: int = -110
    best_spread_away_book: str = ""
    best_over_odds: int = -110
    best_over_book: str = ""
    best_under_odds: int = -110
    best_under_book: str = ""
    best_ml_home_odds: int = 0
    best_ml_home_book: str = ""
    best_ml_away_odds: int = 0
    best_ml_away_book: str = ""


@dataclass
class ValueBet:
    """Identified value betting opportunity."""
    home_team: str
    away_team: str
    bet_type: BetType
    side: str                        # Team name, "over", or "under"

    model_prob: float                # Our predicted probability
    model_line: Optional[float]      # Our predicted line
    market_odds: int                 # Best available odds
    market_line: Optional[float]     # Market line
    market_implied_prob: float       # Vegas implied probability

    edge: float                      # Model prob - market implied prob
    ev: float                        # Expected value per unit bet
    kelly_fraction: float            # Kelly criterion bet size

    recommended_book: str
    confidence: str                  # "high", "medium", "low"

    # Enhanced confidence scoring (optional for backwards compatibility)
    confidence_score: Optional[float] = None  # 0-100 score
    confidence_factors: Optional[Dict[str, Any]] = None  # Detailed breakdown

    @property
    def confidence_breakdown(self) -> str:
        """Human-readable confidence breakdown."""
        if not self.confidence_factors:
            return f"Confidence: {self.confidence.upper()} (based on {self.edge:.1%} edge)"

        factors = self.confidence_factors
        parts = [f"Score: {self.confidence_score:.0f}/100"]

        if "edge" in factors:
            parts.append(f"Edge: {factors['edge'].get('interpretation', 'N/A')}")

        if "statistical" in factors:
            stat = factors["statistical"]
            edges = []
            if stat.get("efficiency_edge"):
                edges.append("efficiency")
            if stat.get("shooting_edge"):
                edges.append("shooting")
            if stat.get("rebounding_edge"):
                edges.append("rebounding")
            if stat.get("turnover_edge"):
                edges.append("turnovers")
            if edges:
                parts.append(f"Statistical edges: {', '.join(edges)}")

        if "model_agreement" in factors:
            ma = factors["model_agreement"]
            if ma.get("score", 0) >= 12:
                parts.append("Models strongly agree")
            elif ma.get("score", 0) >= 7:
                parts.append("Models moderately agree")

        if "variance" in factors:
            if factors["variance"].get("penalty", 0) <= -5:
                parts.append("Warning: High-variance matchup")

        return " | ".join(parts)


@dataclass
class ConfidenceFactors:
    """Detailed breakdown of factors contributing to confidence assessment.

    Scoring breakdown (0-75 points max, scaled to tier):
    - Edge: 0-30 points (raw edge size)
    - Statistical: 0-20 points (team stat advantages)
    - Model agreement: 0-15 points (KenPom vs ML agreement)
    - Situational: 0-10 points (rest, travel, rivalry)
    - Variance penalty: -10 to 0 points
    """
    edge_score: float = 0.0           # Raw edge contribution (0-30 points)
    statistical_edge_score: float = 0.0  # Team stat advantages (0-20 points)
    model_agreement_score: float = 0.0   # KenPom vs ML agreement (0-15 points)
    situational_score: float = 0.0    # Rest, travel, rivalry factors (0-10 points)
    variance_penalty: float = 0.0     # High variance reduces confidence (-10 to 0)

    # Individual factor details for transparency
    factors_detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_score(self) -> float:
        """Total confidence score (0-80 scale, displayed as-is).

        Components:
        - Edge: 0-40 points (PRIMARY factor)
        - Statistical: 0-20 points
        - Model agreement: 7-15 points
        - Situational: 5-10 points
        - Variance penalty: -5 to 0
        """
        raw = (
            self.edge_score +
            self.statistical_edge_score +
            self.model_agreement_score +
            self.situational_score +
            self.variance_penalty
        )
        return max(0, min(80, raw))

    @property
    def confidence_tier(self) -> str:
        """Convert score to tier.

        Edge is now the PRIMARY factor (0-40 points):
        - Edge: 0-40 points (primary - strong edge = high confidence)
        - Statistical: 0-20 points
        - Model agreement: 7-15 points (7 min for single model)
        - Situational: 5-10 points (5 default)
        - Variance penalty: -5 to 0 (reduced impact)

        Max theoretical: 80 points

        Examples with single model + default situational (12 base):
        - 4% edge (~14 pts) + 12 = 26 (Low)
        - 6% edge (~24 pts) + 12 = 36 (Medium)
        - 8% edge (~32 pts) + 12 = 44 (High)
        - 10% edge (~36 pts) + 12 = 48 (High)
        """
        if self.total_score >= 40:
            return "high"
        elif self.total_score >= 30:
            return "medium"
        else:
            return "low"


@dataclass
class SituationalFactors:
    """Game situation factors that affect prediction confidence."""
    home_rest_days: int = 3       # Days since last game
    away_rest_days: int = 3
    home_travel_miles: float = 0  # Miles traveled for this game
    away_travel_miles: float = 0
    is_rivalry_game: bool = False
    is_conference_game: bool = False
    is_tournament: bool = False
    home_recent_form: float = 0.5  # Win rate last 5 games
    away_recent_form: float = 0.5
    season_stage: str = "regular"  # "early", "regular", "late", "tournament"


@dataclass
class PublicBettingData:
    """Public betting percentages and money flow data."""
    # Spread betting
    spread_bet_pct_home: float = 50.0    # % of bets on home spread
    spread_money_pct_home: float = 50.0  # % of money on home spread

    # Total betting
    total_bet_pct_over: float = 50.0     # % of bets on over
    total_money_pct_over: float = 50.0   # % of money on over

    # Moneyline betting
    ml_bet_pct_home: float = 50.0        # % of bets on home ML
    ml_money_pct_home: float = 50.0      # % of money on home ML

    # Ticket counts (optional, for context)
    total_tickets: int = 0               # Total number of tickets

    @property
    def spread_public_side(self) -> str:
        """Which side the public is on for spread."""
        if self.spread_bet_pct_home >= 65:
            return "home_heavy"
        elif self.spread_bet_pct_home <= 35:
            return "away_heavy"
        elif self.spread_bet_pct_home >= 55:
            return "home_slight"
        elif self.spread_bet_pct_home <= 45:
            return "away_slight"
        return "balanced"

    @property
    def sharp_money_indicator(self) -> str:
        """
        Detect sharp money by comparing bet % vs money %.
        If money % significantly differs from bet %, sharps are involved.
        """
        # For spread
        spread_diff = self.spread_money_pct_home - self.spread_bet_pct_home
        if spread_diff >= 10:
            return "sharp_home"  # Sharps on home
        elif spread_diff <= -10:
            return "sharp_away"  # Sharps on away
        return "no_signal"


@dataclass
class BetRecord:
    """Record of a placed bet for CLV tracking."""
    bet_id: str
    timestamp: str                    # ISO format
    game_id: str
    home_team: str
    away_team: str
    bet_type: str                     # "spread", "total", "moneyline"
    side: str                         # Team name, "over", "under"

    # Bet details at time of placement
    placed_line: Optional[float]      # The line when bet was placed
    placed_odds: int                  # American odds when placed
    placed_prob: float                # Our model probability when placed

    # Closing line info (filled in later)
    closing_line: Optional[float] = None
    closing_odds: Optional[int] = None

    # Result (filled in after game)
    result: Optional[str] = None      # "win", "loss", "push"
    actual_margin: Optional[float] = None

    # Derived metrics
    @property
    def clv(self) -> Optional[float]:
        """
        Closing Line Value - difference between placed and closing line.
        Positive CLV = we got a better line than closing (good).
        """
        if self.closing_line is None or self.placed_line is None:
            return None
        # For spreads/totals: line moved in our favor = positive CLV
        return self.closing_line - self.placed_line

    @property
    def clv_cents(self) -> Optional[float]:
        """CLV in 'cents' (half-points)."""
        if self.clv is None:
            return None
        return self.clv * 2  # Each half-point = 1 cent


@dataclass
class CLVSummary:
    """Summary statistics for CLV tracking."""
    total_bets: int = 0
    bets_with_positive_clv: int = 0
    bets_with_negative_clv: int = 0
    avg_clv: float = 0.0
    avg_clv_spread: float = 0.0
    avg_clv_total: float = 0.0
    avg_clv_ml: float = 0.0

    # Win/loss tracking
    total_wins: int = 0
    total_losses: int = 0
    total_pushes: int = 0
    win_rate: float = 0.0

    # CLV correlation with results
    positive_clv_win_rate: float = 0.0
    negative_clv_win_rate: float = 0.0

    # By confidence tier
    high_conf_record: str = "0-0"
    medium_conf_record: str = "0-0"
    low_conf_record: str = "0-0"

    @property
    def clv_edge(self) -> float:
        """Estimated edge based on CLV (rough: 1 point CLV ≈ 3% edge)."""
        return self.avg_clv * 0.03


class CLVTracker:
    """
    Tracks bet records and calculates Closing Line Value (CLV) statistics.

    CLV is one of the best predictors of long-term betting profitability.
    If you consistently beat the closing line, your model has real edge.

    Persists data to JSON file for tracking across sessions.
    """

    def __init__(self, storage_path: str = None):
        """
        Initialize CLV tracker.

        Args:
            storage_path: Path to JSON file for persistence.
                         Defaults to ~/.cbb_betting/clv_records.json
        """
        if storage_path is None:
            home_dir = os.path.expanduser("~")
            storage_dir = os.path.join(home_dir, ".cbb_betting")
            os.makedirs(storage_dir, exist_ok=True)
            storage_path = os.path.join(storage_dir, "clv_records.json")

        self.storage_path = storage_path
        self.records: List[BetRecord] = []
        self._load_records()

    def _load_records(self) -> None:
        """Load records from storage file."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.records = [
                        BetRecord(
                            bet_id=r['bet_id'],
                            timestamp=r['timestamp'],
                            game_id=r['game_id'],
                            home_team=r['home_team'],
                            away_team=r['away_team'],
                            bet_type=r['bet_type'],
                            side=r['side'],
                            placed_line=r.get('placed_line'),
                            placed_odds=r['placed_odds'],
                            placed_prob=r['placed_prob'],
                            closing_line=r.get('closing_line'),
                            closing_odds=r.get('closing_odds'),
                            result=r.get('result'),
                            actual_margin=r.get('actual_margin')
                        )
                        for r in data
                    ]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load CLV records: {e}")
                self.records = []

    def _save_records(self) -> None:
        """Save records to storage file."""
        data = [
            {
                'bet_id': r.bet_id,
                'timestamp': r.timestamp,
                'game_id': r.game_id,
                'home_team': r.home_team,
                'away_team': r.away_team,
                'bet_type': r.bet_type,
                'side': r.side,
                'placed_line': r.placed_line,
                'placed_odds': r.placed_odds,
                'placed_prob': r.placed_prob,
                'closing_line': r.closing_line,
                'closing_odds': r.closing_odds,
                'result': r.result,
                'actual_margin': r.actual_margin
            }
            for r in self.records
        ]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def record_bet(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        bet_type: str,
        side: str,
        placed_line: Optional[float],
        placed_odds: int,
        placed_prob: float,
        confidence_tier: str = "medium"
    ) -> BetRecord:
        """
        Record a new bet placement.

        Args:
            game_id: Unique identifier for the game
            home_team: Home team name
            away_team: Away team name
            bet_type: "spread", "total", or "moneyline"
            side: Team name for spread/ML, "over"/"under" for totals
            placed_line: The line when bet was placed (None for ML)
            placed_odds: American odds when placed
            placed_prob: Our model's probability when placed
            confidence_tier: "high", "medium", or "low"

        Returns:
            The created BetRecord
        """
        bet_id = f"{game_id}_{bet_type}_{side}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        record = BetRecord(
            bet_id=bet_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            bet_type=bet_type,
            side=side,
            placed_line=placed_line,
            placed_odds=placed_odds,
            placed_prob=placed_prob
        )

        self.records.append(record)
        self._save_records()
        return record

    def update_closing_line(
        self,
        bet_id: str,
        closing_line: Optional[float],
        closing_odds: Optional[int] = None
    ) -> Optional[BetRecord]:
        """
        Update a bet record with closing line information.

        Args:
            bet_id: ID of the bet to update
            closing_line: The closing line
            closing_odds: The closing odds (optional)

        Returns:
            Updated BetRecord or None if not found
        """
        for record in self.records:
            if record.bet_id == bet_id:
                record.closing_line = closing_line
                if closing_odds is not None:
                    record.closing_odds = closing_odds
                self._save_records()
                return record
        return None

    def record_result(
        self,
        bet_id: str,
        result: str,
        actual_margin: Optional[float] = None
    ) -> Optional[BetRecord]:
        """
        Record the result of a bet.

        Args:
            bet_id: ID of the bet to update
            result: "win", "loss", or "push"
            actual_margin: Actual margin/total for the game

        Returns:
            Updated BetRecord or None if not found
        """
        for record in self.records:
            if record.bet_id == bet_id:
                record.result = result
                record.actual_margin = actual_margin
                self._save_records()
                return record
        return None

    def update_game_results(
        self,
        game_id: str,
        home_score: int,
        away_score: int,
        closing_spread: Optional[float] = None,
        closing_total: Optional[float] = None
    ) -> List[BetRecord]:
        """
        Update all bets for a game with final results.

        Args:
            game_id: Game identifier
            home_score: Final home team score
            away_score: Final away team score
            closing_spread: Closing spread line
            closing_total: Closing total line

        Returns:
            List of updated records
        """
        actual_margin = home_score - away_score
        actual_total = home_score + away_score
        updated = []

        for record in self.records:
            if record.game_id == game_id:
                # Update closing lines if provided
                if record.bet_type == "spread" and closing_spread is not None:
                    record.closing_line = closing_spread
                elif record.bet_type == "total" and closing_total is not None:
                    record.closing_line = closing_total

                # Determine result
                if record.bet_type == "spread":
                    # Check if bet covered
                    if record.side == record.home_team:
                        # Home spread bet - need to beat the spread
                        spread_line = record.placed_line or 0
                        cover_margin = actual_margin + spread_line
                        if cover_margin > 0:
                            record.result = "win"
                        elif cover_margin < 0:
                            record.result = "loss"
                        else:
                            record.result = "push"
                    else:
                        # Away spread bet
                        spread_line = -(record.placed_line or 0)
                        cover_margin = -actual_margin + spread_line
                        if cover_margin > 0:
                            record.result = "win"
                        elif cover_margin < 0:
                            record.result = "loss"
                        else:
                            record.result = "push"
                    record.actual_margin = actual_margin

                elif record.bet_type == "total":
                    total_line = record.placed_line or 0
                    if record.side == "over":
                        if actual_total > total_line:
                            record.result = "win"
                        elif actual_total < total_line:
                            record.result = "loss"
                        else:
                            record.result = "push"
                    else:  # under
                        if actual_total < total_line:
                            record.result = "win"
                        elif actual_total > total_line:
                            record.result = "loss"
                        else:
                            record.result = "push"
                    record.actual_margin = actual_total

                elif record.bet_type == "moneyline":
                    if record.side == record.home_team:
                        record.result = "win" if actual_margin > 0 else "loss" if actual_margin < 0 else "push"
                    else:
                        record.result = "win" if actual_margin < 0 else "loss" if actual_margin > 0 else "push"
                    record.actual_margin = actual_margin

                updated.append(record)

        if updated:
            self._save_records()

        return updated

    def get_summary(
        self,
        bet_type: Optional[str] = None,
        days: Optional[int] = None
    ) -> CLVSummary:
        """
        Get CLV summary statistics.

        Args:
            bet_type: Filter by bet type ("spread", "total", "moneyline")
            days: Only include bets from last N days

        Returns:
            CLVSummary with statistics
        """
        filtered = self.records

        # Filter by bet type
        if bet_type:
            filtered = [r for r in filtered if r.bet_type == bet_type]

        # Filter by date
        if days:
            cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
            filtered = [
                r for r in filtered
                if datetime.fromisoformat(r.timestamp).timestamp() >= cutoff
            ]

        if not filtered:
            return CLVSummary()

        # Calculate CLV stats
        clv_values = [r.clv for r in filtered if r.clv is not None]
        spread_clv = [r.clv for r in filtered if r.bet_type == "spread" and r.clv is not None]
        total_clv = [r.clv for r in filtered if r.bet_type == "total" and r.clv is not None]
        ml_clv = [r.clv for r in filtered if r.bet_type == "moneyline" and r.clv is not None]

        # Calculate win/loss stats
        completed = [r for r in filtered if r.result is not None]
        wins = len([r for r in completed if r.result == "win"])
        losses = len([r for r in completed if r.result == "loss"])
        pushes = len([r for r in completed if r.result == "push"])

        # CLV correlation with results
        positive_clv_bets = [r for r in completed if r.clv is not None and r.clv > 0]
        negative_clv_bets = [r for r in completed if r.clv is not None and r.clv < 0]

        pos_clv_wins = len([r for r in positive_clv_bets if r.result == "win"])
        neg_clv_wins = len([r for r in negative_clv_bets if r.result == "win"])

        summary = CLVSummary(
            total_bets=len(filtered),
            bets_with_positive_clv=len([v for v in clv_values if v > 0]),
            bets_with_negative_clv=len([v for v in clv_values if v < 0]),
            avg_clv=sum(clv_values) / len(clv_values) if clv_values else 0.0,
            avg_clv_spread=sum(spread_clv) / len(spread_clv) if spread_clv else 0.0,
            avg_clv_total=sum(total_clv) / len(total_clv) if total_clv else 0.0,
            avg_clv_ml=sum(ml_clv) / len(ml_clv) if ml_clv else 0.0,
            total_wins=wins,
            total_losses=losses,
            total_pushes=pushes,
            win_rate=wins / (wins + losses) if (wins + losses) > 0 else 0.0,
            positive_clv_win_rate=pos_clv_wins / len(positive_clv_bets) if positive_clv_bets else 0.0,
            negative_clv_win_rate=neg_clv_wins / len(negative_clv_bets) if negative_clv_bets else 0.0
        )

        return summary

    def get_records_by_game(self, game_id: str) -> List[BetRecord]:
        """Get all bet records for a specific game."""
        return [r for r in self.records if r.game_id == game_id]

    def get_pending_results(self) -> List[BetRecord]:
        """Get bets that don't have results yet."""
        return [r for r in self.records if r.result is None]

    def get_pending_closing_lines(self) -> List[BetRecord]:
        """Get bets that don't have closing lines yet."""
        return [r for r in self.records if r.closing_line is None]

    def clear_old_records(self, days: int = 90) -> int:
        """
        Remove records older than specified days.

        Args:
            days: Remove records older than this

        Returns:
            Number of records removed
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
        original_count = len(self.records)
        self.records = [
            r for r in self.records
            if datetime.fromisoformat(r.timestamp).timestamp() >= cutoff
        ]
        removed = original_count - len(self.records)
        if removed > 0:
            self._save_records()
        return removed

    def export_to_csv(self, output_path: str) -> None:
        """
        Export records to CSV for analysis.

        Args:
            output_path: Path to output CSV file
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'bet_id', 'timestamp', 'game_id', 'home_team', 'away_team',
                'bet_type', 'side', 'placed_line', 'placed_odds', 'placed_prob',
                'closing_line', 'closing_odds', 'clv', 'result', 'actual_margin'
            ])
            for r in self.records:
                writer.writerow([
                    r.bet_id, r.timestamp, r.game_id, r.home_team, r.away_team,
                    r.bet_type, r.side, r.placed_line, r.placed_odds, r.placed_prob,
                    r.closing_line, r.closing_odds, r.clv, r.result, r.actual_margin
                ])


class ValueCalculator:
    """
    Calculate value bets by comparing model predictions to market odds.

    Uses advanced KenPom metrics, situational factors, and multi-factor
    confidence scoring for more accurate value detection.
    """

    # Minimum edge required to flag as a value bet
    MIN_EDGE_SPREAD = 0.03    # 3% edge on spread
    MIN_EDGE_TOTAL = 0.03     # 3% edge on totals
    MIN_EDGE_ML = 0.05        # 5% edge on moneyline

    # Standard deviation for spread outcomes (typical for college basketball)
    BASE_SPREAD_STD_DEV = 10.0
    BASE_TOTAL_STD_DEV = 12.0

    # Kelly fraction multiplier (fractional Kelly is safer)
    KELLY_FRACTION = 0.25     # Use 1/4 Kelly

    # Adjustment factors for advanced metrics
    TEMPO_MISMATCH_THRESHOLD = 5.0    # Significant tempo difference
    THREE_PT_VOLATILITY_WEIGHT = 0.3  # How much 3pt reliance increases variance
    EXPERIENCE_FACTOR = 0.15          # How much experience reduces variance

    # Confidence scoring thresholds
    EDGE_THRESHOLDS = {
        "spread": {"excellent": 0.10, "good": 0.06, "fair": 0.04},
        "total": {"excellent": 0.10, "good": 0.06, "fair": 0.04},
        "moneyline": {"excellent": 0.15, "good": 0.10, "fair": 0.06}
    }

    def __init__(
        self,
        min_edge_spread: float = 0.03,
        min_edge_total: float = 0.03,
        min_edge_ml: float = 0.05
    ):
        self.min_edge_spread = min_edge_spread
        self.min_edge_total = min_edge_total
        self.min_edge_ml = min_edge_ml

    # ==================== ADVANCED METRICS ====================

    def calculate_adjusted_spread_std_dev(
        self,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats]
    ) -> float:
        """
        Calculate game-specific standard deviation for spread based on team characteristics.

        Higher variance games (higher std dev) include:
        - Teams that rely heavily on 3-point shooting (more volatile)
        - Tempo mismatches (unpredictable game flow)
        - Low experience teams (less consistent)

        Lower variance games (lower std dev) include:
        - Experienced teams
        - Interior-focused offenses
        - Similar tempo teams
        """
        std_dev = self.BASE_SPREAD_STD_DEV

        if not home_stats or not away_stats:
            return std_dev

        # Adjust for 3-point reliance (more 3s = more variance)
        avg_3pt_reliance = (
            home_stats.three_point_reliance + away_stats.three_point_reliance
        ) / 2
        std_dev += avg_3pt_reliance * self.THREE_PT_VOLATILITY_WEIGHT * 3.0

        # Adjust for experience (more experience = less variance)
        avg_experience = (home_stats.experience + away_stats.experience) / 2
        if avg_experience > 0:
            # Experience typically ranges from 0.5 to 2.5 years
            exp_factor = min(avg_experience / 2.0, 1.0)  # Cap at 1.0
            std_dev -= exp_factor * self.EXPERIENCE_FACTOR * 2.0

        # Adjust for tempo mismatch (big difference = more unpredictable)
        tempo_diff = abs(home_stats.adj_tempo - away_stats.adj_tempo)
        if tempo_diff > self.TEMPO_MISMATCH_THRESHOLD:
            std_dev += (tempo_diff - self.TEMPO_MISMATCH_THRESHOLD) * 0.2

        # Ensure std_dev stays in reasonable range
        return max(8.0, min(14.0, std_dev))

    def calculate_adjusted_total_std_dev(
        self,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats],
        predicted_tempo: float = 0
    ) -> float:
        """
        Calculate game-specific standard deviation for totals.

        Factors that affect total variance:
        - Game tempo (faster pace = more variance in total)
        - 3-point shooting reliance
        - Defensive consistency
        - NEW: Average Possession Length (APL) data
        """
        std_dev = self.BASE_TOTAL_STD_DEV

        if not home_stats or not away_stats:
            return std_dev

        # Adjust for average tempo
        if predicted_tempo > 0:
            # Average tempo is around 67-68, faster games have more variance
            tempo_adjustment = (predicted_tempo - 67) * 0.15
            std_dev += tempo_adjustment

        # 3-point shooting increases total variance
        avg_3pt_reliance = (
            home_stats.three_point_reliance + away_stats.three_point_reliance
        ) / 2
        std_dev += avg_3pt_reliance * 4.0  # Up to ~1.5 points extra std dev

        # NEW: Use APL data for more accurate pace predictions
        # Teams with extreme pace factors (very fast or very slow) tend to be
        # more predictable on totals - reduce std dev
        home_pace = home_stats.pace_factor
        away_pace = away_stats.pace_factor
        if home_pace > 0 and away_pace > 0:
            avg_pace = (home_pace + away_pace) / 2
            # Extreme paces (both teams fast or both slow) = more predictable
            pace_deviation = abs(avg_pace - 1.0)
            if pace_deviation > 0.05:
                # Reduce variance for more extreme paces (more predictable)
                std_dev -= pace_deviation * 3.0

        return max(9.0, min(16.0, std_dev))

    def calculate_matchup_edge_boost(
        self,
        stat_comparison: Optional['TeamStatComparison']
    ) -> float:
        """
        Calculate additional edge based on statistical matchup advantages.

        Returns a multiplier (0.9 to 1.2) to boost or reduce confidence.

        NEW: Uses SOS, Pythag, and trend data for better accuracy.
        """
        if not stat_comparison:
            return 1.0

        boost = 1.0
        home_stats = stat_comparison.home_stats
        away_stats = stat_comparison.away_stats

        # Major efficiency edge increases confidence
        if stat_comparison.efficiency_edge:
            boost += 0.05

        # Major shooting edge
        if stat_comparison.shooting_edge:
            boost += 0.03

        # Major rebounding edge
        if stat_comparison.rebounding_edge:
            boost += 0.02

        # Major turnover edge
        if stat_comparison.turnover_edge:
            boost += 0.02

        # Tempo mismatch reduces confidence slightly (more unpredictable)
        if stat_comparison.tempo_mismatch:
            boost -= 0.05

        # NEW: SOS-based confidence adjustments
        # Both teams with strong SOS = more reliable prediction
        if home_stats and away_stats:
            home_sos_tier = home_stats.schedule_strength_tier
            away_sos_tier = away_stats.schedule_strength_tier

            if home_sos_tier == "elite" and away_sos_tier == "elite":
                boost += 0.04  # Both faced tough competition, ratings reliable
            elif home_sos_tier == "elite" or away_sos_tier == "elite":
                boost += 0.02  # At least one has elite SOS
            elif home_sos_tier == "weak" and away_sos_tier == "weak":
                boost -= 0.03  # Both have weak SOS, ratings less reliable

            # NEW: Pythag-based reliability
            # Teams with Pythag close to their record are more predictable
            if home_stats.pythag > 0 and away_stats.pythag > 0:
                # High Pythag teams (>0.85) against low Pythag (<0.4) = higher confidence
                pythag_diff = abs(home_stats.pythag - away_stats.pythag)
                if pythag_diff > 0.3:
                    boost += 0.03  # Large skill gap = more predictable outcome

            # NEW: Trend-based adjustments
            # Trending teams = less reliable (ratings may not reflect current form)
            home_trending = home_stats.is_trending_up or home_stats.is_trending_down
            away_trending = away_stats.is_trending_up or away_stats.is_trending_down

            if home_trending and away_trending:
                boost -= 0.04  # Both teams in flux, less predictable
            elif home_trending or away_trending:
                boost -= 0.02  # One team trending, slight uncertainty

            # Confidence boost if trending team is trending in our favor
            # (e.g., betting on a team that's getting better)
            if home_stats.is_trending_up and home_stats.adj_em_trend > 2.0:
                boost += 0.02  # Strong upward trend for home team
            if away_stats.is_trending_up and away_stats.adj_em_trend > 2.0:
                boost += 0.02  # Strong upward trend for away team

        return max(0.85, min(1.25, boost))

    def calculate_pythag_adjusted_win_prob(
        self,
        kenpom_win_prob: float,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats]
    ) -> float:
        """
        Adjust KenPom win probability using Pythagorean ratings.

        Pythag ratings are often more reliable than simple efficiency margins
        because they account for actual game outcomes. This method blends
        the KenPom prediction with Pythag-based expectations.

        Args:
            kenpom_win_prob: KenPom's predicted win probability for home team
            home_stats: Home team statistics including Pythag
            away_stats: Away team statistics including Pythag

        Returns:
            Adjusted win probability for home team
        """
        if not home_stats or not away_stats:
            return kenpom_win_prob

        home_pythag = home_stats.pythag
        away_pythag = away_stats.pythag

        # If Pythag data not available, return original
        if home_pythag <= 0 or away_pythag <= 0:
            return kenpom_win_prob

        # Calculate Pythag-based win probability
        # Using log5 method: P(A beats B) = (pA - pA*pB) / (pA + pB - 2*pA*pB)
        # Where pA, pB are Pythag winning percentages
        numerator = home_pythag - (home_pythag * away_pythag)
        denominator = home_pythag + away_pythag - (2 * home_pythag * away_pythag)

        if denominator <= 0:
            pythag_win_prob = 0.5
        else:
            pythag_win_prob = numerator / denominator

        # Add home court advantage (~3.5 points in college = ~0.04 win prob)
        pythag_win_prob = min(0.99, pythag_win_prob + 0.04)

        # Blend KenPom and Pythag predictions (60% KenPom, 40% Pythag)
        # KenPom gets more weight as it accounts for opponent-adjusted metrics
        blended_prob = (kenpom_win_prob * 0.60) + (pythag_win_prob * 0.40)

        return max(0.01, min(0.99, blended_prob))

    def calculate_total_adjustment(
        self,
        kenpom_total: float,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats],
        predicted_tempo: float = 0
    ) -> float:
        """
        Adjust predicted total using APL (Average Possession Length) data.

        APL provides insight into how teams actually use shot clock time,
        which affects the number of possessions and thus total points.

        Args:
            kenpom_total: KenPom's predicted total
            home_stats: Home team statistics including APL
            away_stats: Away team statistics including APL
            predicted_tempo: KenPom's predicted tempo

        Returns:
            Adjusted total prediction
        """
        if not home_stats or not away_stats:
            return kenpom_total

        # Get pace factors from APL data
        home_pace = home_stats.pace_factor
        away_pace = away_stats.pace_factor

        if home_pace <= 0 or away_pace <= 0:
            return kenpom_total

        # Average pace factor for the game
        avg_pace = (home_pace + away_pace) / 2

        # Adjust total based on pace deviation from average
        # If pace factor > 1, teams play faster = more possessions = higher total
        # Each 0.05 pace deviation = roughly 2 points adjustment
        pace_adjustment = (avg_pace - 1.0) * 40  # Scale factor

        # Apply smaller adjustment (APL is already somewhat reflected in tempo)
        adjusted_total = kenpom_total + (pace_adjustment * 0.3)

        return max(100, min(200, adjusted_total))  # Reasonable bounds

    # ==================== ODDS CONVERSION ====================
    
    @staticmethod
    def american_to_implied_prob(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    @staticmethod
    def implied_prob_to_american(prob: float) -> int:
        """Convert implied probability to American odds."""
        if prob <= 0 or prob >= 1:
            return 0
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)
    
    @staticmethod
    def american_to_decimal(odds: int) -> float:
        """Convert American odds to decimal odds."""
        if odds == 0:
            return 0.0  # No odds available
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    
    # ==================== PROBABILITY CALCULATIONS ====================
    
    def spread_to_win_prob(self, spread: float, std_dev: float = None) -> float:
        """
        Convert a predicted spread to win probability using normal distribution.
        
        Args:
            spread: Predicted margin (positive = team A wins by that amount)
            std_dev: Standard deviation of outcomes
            
        Returns:
            Probability of team A winning
        """
        if std_dev is None:
            std_dev = self.BASE_SPREAD_STD_DEV
        
        # Using CDF of normal distribution
        # P(win) = P(margin > 0) = 1 - CDF(-spread/std_dev)
        z = spread / std_dev
        return self._norm_cdf(z)
    
    def cover_spread_prob(
        self,
        predicted_margin: float,
        line: float,
        std_dev: float = None
    ) -> float:
        """
        Calculate probability of covering a spread.

        Args:
            predicted_margin: Our predicted margin (positive = home team wins by that much)
            line: The spread line (e.g., -6.5 means home favored by 6.5, must win by >6.5)
            std_dev: Standard deviation of outcomes

        Returns:
            Probability of home team covering
        """
        if std_dev is None:
            std_dev = self.BASE_SPREAD_STD_DEV

        # Home team covers if actual_margin > -line (beat the spread)
        # E.g., if line = -6.5 (home -6.5), home covers if they win by > 6.5
        # P(margin > -line) = P(margin - predicted_margin > -line - predicted_margin)
        # = 1 - CDF((-line - predicted_margin) / std_dev)
        # = CDF((predicted_margin + line) / std_dev)

        z = (predicted_margin + line) / std_dev
        return self._norm_cdf(z)
    
    def over_under_prob(
        self,
        predicted_total: float,
        line: float,
        std_dev: float = 12.0
    ) -> Tuple[float, float]:
        """
        Calculate probability of over/under.
        
        Args:
            predicted_total: Our predicted total
            line: The total line
            std_dev: Standard deviation of total outcomes
            
        Returns:
            Tuple of (over_prob, under_prob)
        """
        z = (predicted_total - line) / std_dev
        over_prob = self._norm_cdf(z)
        under_prob = 1 - over_prob
        return over_prob, under_prob
    
    @staticmethod
    def _norm_cdf(z: float) -> float:
        """Cumulative distribution function for standard normal."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    
    # ==================== VALUE CALCULATION ====================
    
    def calculate_ev(self, prob: float, odds: int) -> float:
        """
        Calculate expected value of a bet.

        Args:
            prob: Our estimated probability of winning
            odds: American odds

        Returns:
            Expected value per unit wagered
        """
        if odds == 0:
            return -1.0  # No odds available, return negative EV
        decimal_odds = self.american_to_decimal(odds)
        return (prob * decimal_odds) - 1
    
    def calculate_kelly(self, prob: float, odds: int) -> float:
        """
        Calculate Kelly criterion bet size.
        
        Args:
            prob: Our estimated probability of winning
            odds: American odds
            
        Returns:
            Recommended bet size as fraction of bankroll
        """
        decimal_odds = self.american_to_decimal(odds)
        q = 1 - prob
        b = decimal_odds - 1
        
        kelly = (prob * b - q) / b
        
        # Apply fractional Kelly and cap at reasonable maximum
        kelly = max(0, kelly * self.KELLY_FRACTION)
        return min(kelly, 0.10)  # Cap at 10% of bankroll
    
    def get_confidence_level(self, edge: float, bet_type: BetType) -> str:
        """Categorize confidence based on edge size (legacy method for backwards compatibility)."""
        if bet_type == BetType.MONEYLINE:
            if edge >= 0.15:
                return "high"
            elif edge >= 0.08:
                return "medium"
            else:
                return "low"
        else:
            if edge >= 0.08:
                return "high"
            elif edge >= 0.05:
                return "medium"
            else:
                return "low"

    def calculate_confidence_score(
        self,
        edge: float,
        bet_type: BetType,
        model_prob: float,
        market_implied_prob: float,
        stat_comparison: Optional[TeamStatComparison] = None,
        kenpom_prediction: Optional[float] = None,
        ml_prediction: Optional[float] = None,
        situational: Optional[SituationalFactors] = None,
        line_movement: Optional[Dict[str, float]] = None,
        home_stats: Optional[TeamStats] = None,
        away_stats: Optional[TeamStats] = None,
        **kwargs  # Accept but ignore unused params for backwards compatibility
    ) -> ConfidenceFactors:
        """
        Calculate comprehensive confidence score using multiple factors.

        This multi-factor approach provides more accurate value assessment than
        simple edge thresholds by considering:
        1. Raw edge size and EV (0-30 points)
        2. Statistical matchup advantages (0-20 points)
        3. Model agreement (KenPom vs ML) (0-15 points)
        4. Situational factors (0-10 points)
        5. Game variance characteristics (-10 to 0 penalty)

        Max score: 75 points
        High: >= 45, Medium: >= 30, Low: < 30

        Returns:
            ConfidenceFactors with detailed scoring breakdown
        """
        factors = ConfidenceFactors(factors_detail={})

        # 1. EDGE SCORE (0-30 points)
        # Larger edges indicate stronger value opportunities
        factors.edge_score = self._score_edge(edge, bet_type)
        factors.factors_detail["edge"] = {
            "raw_edge": edge,
            "score": factors.edge_score,
            "interpretation": self._interpret_edge(edge, bet_type)
        }

        # 2. STATISTICAL EDGE SCORE (0-20 points)
        # Teams with clear statistical advantages are more predictable
        if stat_comparison:
            factors.statistical_edge_score = self._score_statistical_matchup(
                stat_comparison, bet_type
            )
            factors.factors_detail["statistical"] = {
                "score": factors.statistical_edge_score,
                "efficiency_edge": stat_comparison.efficiency_edge,
                "shooting_edge": stat_comparison.shooting_edge,
                "rebounding_edge": stat_comparison.rebounding_edge,
                "turnover_edge": stat_comparison.turnover_edge,
                "major_differences": len(stat_comparison.major_differences)
            }

        # 3. MODEL AGREEMENT SCORE (0-15 points)
        # When multiple models agree, confidence increases
        if kenpom_prediction is not None and ml_prediction is not None:
            factors.model_agreement_score = self._score_model_agreement(
                kenpom_prediction, ml_prediction, market_implied_prob, bet_type
            )
            factors.factors_detail["model_agreement"] = {
                "score": factors.model_agreement_score,
                "kenpom_pred": kenpom_prediction,
                "ml_pred": ml_prediction,
                "market_implied": market_implied_prob
            }
        else:
            # Single source - partial credit for having any model
            factors.model_agreement_score = 7.0
            factors.factors_detail["model_agreement"] = {
                "score": 7.0,
                "note": "Single model prediction"
            }

        # 4. SITUATIONAL SCORE (0-10 points)
        # Rest, travel, and game context affect predictability
        if situational:
            factors.situational_score = self._score_situational_factors(
                situational, bet_type
            )
            factors.factors_detail["situational"] = {
                "score": factors.situational_score,
                "home_rest": situational.home_rest_days,
                "away_rest": situational.away_rest_days,
                "rivalry": situational.is_rivalry_game,
                "conference": situational.is_conference_game,
                "season_stage": situational.season_stage
            }
        else:
            # Default neutral situational score
            factors.situational_score = 5.0
            factors.factors_detail["situational"] = {
                "score": 5.0,
                "note": "Standard game conditions assumed"
            }

        # 5. VARIANCE PENALTY (-10 to 0 points)
        # High-variance games reduce confidence in any prediction
        factors.variance_penalty = self._calculate_variance_penalty(
            home_stats, away_stats, bet_type, stat_comparison
        )
        factors.factors_detail["variance"] = {
            "penalty": factors.variance_penalty,
            "note": "Negative values indicate high-variance matchup"
        }

        return factors

    def _score_public_betting(
        self,
        public_data: 'PublicBettingData',
        bet_type: BetType,
        bet_side: str
    ) -> float:
        """
        Score based on public betting percentages (0-15 points).

        Key principle: Fading heavy public action tends to be profitable.
        Sharp money (money % differs from bet %) is a strong signal.
        """
        score = 5.0  # Start neutral

        if bet_type == BetType.SPREAD:
            bet_pct = public_data.spread_bet_pct_home if bet_side == "home" else (100 - public_data.spread_bet_pct_home)
            money_pct = public_data.spread_money_pct_home if bet_side == "home" else (100 - public_data.spread_money_pct_home)
        elif bet_type == BetType.TOTAL:
            bet_pct = public_data.total_bet_pct_over if bet_side == "over" else (100 - public_data.total_bet_pct_over)
            money_pct = public_data.total_money_pct_over if bet_side == "over" else (100 - public_data.total_money_pct_over)
        else:  # MONEYLINE
            bet_pct = public_data.ml_bet_pct_home if bet_side == "home" else (100 - public_data.ml_bet_pct_home)
            money_pct = public_data.ml_money_pct_home if bet_side == "home" else (100 - public_data.ml_money_pct_home)

        # FADE THE PUBLIC (0-8 points)
        # Betting against heavy public action is historically profitable
        if bet_pct <= 30:
            # We're taking the contrarian side (public < 30% on our side)
            score += 8.0  # Strong contrarian play
        elif bet_pct <= 40:
            score += 5.0  # Moderate contrarian
        elif bet_pct <= 50:
            score += 2.0  # Slight contrarian
        elif bet_pct >= 70:
            # We're with the heavy public - be cautious
            score -= 3.0
        elif bet_pct >= 60:
            score -= 1.0

        # SHARP MONEY INDICATOR (0-7 points)
        # When money % significantly exceeds bet %, sharps are involved
        money_vs_bet_diff = money_pct - bet_pct

        if money_vs_bet_diff >= 15:
            # Heavy sharp action on our side
            score += 7.0
        elif money_vs_bet_diff >= 10:
            score += 5.0
        elif money_vs_bet_diff >= 5:
            score += 3.0
        elif money_vs_bet_diff <= -15:
            # Sharp money against us - warning
            score -= 4.0
        elif money_vs_bet_diff <= -10:
            score -= 2.0

        return max(0, min(15, score))

    def _get_public_pct(
        self,
        public_data: 'PublicBettingData',
        bet_type: BetType,
        bet_side: str
    ) -> float:
        """Get public bet percentage for our side."""
        if bet_type == BetType.SPREAD:
            return public_data.spread_bet_pct_home if bet_side == "home" else (100 - public_data.spread_bet_pct_home)
        elif bet_type == BetType.TOTAL:
            return public_data.total_bet_pct_over if bet_side == "over" else (100 - public_data.total_bet_pct_over)
        else:
            return public_data.ml_bet_pct_home if bet_side == "home" else (100 - public_data.ml_bet_pct_home)

    def _get_money_pct(
        self,
        public_data: 'PublicBettingData',
        bet_type: BetType,
        bet_side: str
    ) -> float:
        """Get money percentage for our side."""
        if bet_type == BetType.SPREAD:
            return public_data.spread_money_pct_home if bet_side == "home" else (100 - public_data.spread_money_pct_home)
        elif bet_type == BetType.TOTAL:
            return public_data.total_money_pct_over if bet_side == "over" else (100 - public_data.total_money_pct_over)
        else:
            return public_data.ml_money_pct_home if bet_side == "home" else (100 - public_data.ml_money_pct_home)

    def _score_historical_accuracy(
        self,
        clv_summary: 'CLVSummary',
        bet_type: BetType
    ) -> float:
        """
        Score based on historical model accuracy (0-10 points).

        CLV (Closing Line Value) is the best predictor of long-term profitability.
        If we consistently beat the closing line, our model has edge.
        """
        score = 5.0  # Start neutral

        if clv_summary.total_bets < 10:
            # Not enough data to judge
            return 5.0

        # CLV-based scoring (0-5 points)
        # Positive CLV = beating closing line = good
        avg_clv = clv_summary.avg_clv
        if bet_type == BetType.SPREAD or bet_type == BetType.TOTAL:
            avg_clv = clv_summary.avg_clv_spread if bet_type == BetType.SPREAD else clv_summary.avg_clv_total
        else:
            avg_clv = clv_summary.avg_clv_ml

        if avg_clv >= 1.0:
            # Excellent: averaging 1+ point CLV
            score += 5.0
        elif avg_clv >= 0.5:
            score += 3.0
        elif avg_clv >= 0.0:
            score += 1.0
        elif avg_clv <= -1.0:
            score -= 3.0
        elif avg_clv <= -0.5:
            score -= 1.0

        # Win rate adjustment (0-3 points)
        if clv_summary.win_rate >= 0.55:
            score += 3.0
        elif clv_summary.win_rate >= 0.52:
            score += 2.0
        elif clv_summary.win_rate >= 0.50:
            score += 1.0
        elif clv_summary.win_rate < 0.45:
            score -= 2.0

        # Positive CLV correlation (0-2 points)
        # Bets with positive CLV should win more often
        if clv_summary.positive_clv_win_rate >= 0.55:
            score += 2.0
        elif clv_summary.positive_clv_win_rate >= 0.52:
            score += 1.0

        return max(0, min(10, score))

    def _score_edge(self, edge: float, bet_type: BetType) -> float:
        """Score the raw edge (0-40 points).

        Edge is the PRIMARY indicator of value. Higher weight than other factors.
        A strong edge (8%+) should put bets into high confidence territory.
        """
        type_key = bet_type.value if bet_type != BetType.MONEYLINE else "moneyline"
        thresholds = self.EDGE_THRESHOLDS.get(type_key, self.EDGE_THRESHOLDS["spread"])

        if edge >= thresholds["excellent"]:
            # Excellent edge (8%+): 32-40 points
            excess = edge - thresholds["excellent"]
            return min(40, 32 + (excess / 0.05) * 8)
        elif edge >= thresholds["good"]:
            # Good edge (6-8%): 24-31 points
            ratio = (edge - thresholds["good"]) / (thresholds["excellent"] - thresholds["good"])
            return 24 + ratio * 7
        elif edge >= thresholds["fair"]:
            # Fair edge (4-6%): 14-23 points
            ratio = (edge - thresholds["fair"]) / (thresholds["good"] - thresholds["fair"])
            return 14 + ratio * 9
        else:
            # Marginal edge (<4%): 0-13 points
            ratio = edge / thresholds["fair"]
            return ratio * 13

    def _interpret_edge(self, edge: float, bet_type: BetType) -> str:
        """Provide human-readable interpretation of edge."""
        type_key = bet_type.value if bet_type != BetType.MONEYLINE else "moneyline"
        thresholds = self.EDGE_THRESHOLDS.get(type_key, self.EDGE_THRESHOLDS["spread"])

        if edge >= thresholds["excellent"]:
            return "excellent - strong value opportunity"
        elif edge >= thresholds["good"]:
            return "good - solid value"
        elif edge >= thresholds["fair"]:
            return "fair - marginal value"
        else:
            return "minimal - borderline value"

    def _score_statistical_matchup(
        self,
        stat_comparison: TeamStatComparison,
        bet_type: BetType
    ) -> float:
        """Score based on statistical advantages (0-20 points)."""
        score = 0.0

        # Count major statistical edges and their relevance to bet type
        major_diffs = stat_comparison.major_differences

        # Efficiency edge is most important (0-8 points)
        if stat_comparison.efficiency_edge:
            score += 8.0

        # Shooting edge matters for all bets (0-4 points)
        if stat_comparison.shooting_edge:
            score += 4.0

        # Rebounding matters more for totals (0-3 points)
        if stat_comparison.rebounding_edge:
            if bet_type == BetType.TOTAL:
                score += 3.0
            else:
                score += 2.0

        # Turnover edge (0-3 points)
        if stat_comparison.turnover_edge:
            score += 3.0

        # Bonus for multiple aligned edges (0-2 points)
        edge_count = sum([
            bool(stat_comparison.efficiency_edge),
            bool(stat_comparison.shooting_edge),
            bool(stat_comparison.rebounding_edge),
            bool(stat_comparison.turnover_edge)
        ])
        if edge_count >= 3:
            score += 2.0

        # Penalty for tempo mismatch (reduces predictability)
        if stat_comparison.tempo_mismatch:
            score -= 2.0

        return max(0, min(20, score))

    def _score_model_agreement(
        self,
        kenpom_pred: float,
        ml_pred: float,
        market_implied: float,
        bet_type: BetType
    ) -> float:
        """Score based on model agreement (0-15 points)."""
        score = 0.0

        # Calculate agreement between KenPom and ML predictions
        pred_diff = abs(kenpom_pred - ml_pred)

        # Strong agreement (within 3%): 10-15 points
        if pred_diff <= 0.03:
            score = 12.0
            # Bonus if both significantly disagree with market
            avg_pred = (kenpom_pred + ml_pred) / 2
            market_diff = abs(avg_pred - market_implied)
            if market_diff >= 0.05:
                score = 15.0
        # Moderate agreement (within 6%): 6-9 points
        elif pred_diff <= 0.06:
            score = 7.0
            # Some bonus if direction agrees vs market
            if (kenpom_pred > market_implied) == (ml_pred > market_implied):
                score = 9.0
        # Weak agreement (within 10%): 3-5 points
        elif pred_diff <= 0.10:
            score = 4.0
        # Disagreement: 0-2 points
        else:
            score = 1.0

        return score

    def _score_market_inefficiency(
        self,
        model_prob: float,
        market_implied: float,
        line_movement: Optional[Dict[str, float]] = None
    ) -> float:
        """Score based on market inefficiency signals (0-15 points)."""
        score = 0.0
        prob_gap = model_prob - market_implied

        # Large probability gap suggests market inefficiency (0-8 points)
        abs_gap = abs(prob_gap)
        if abs_gap >= 0.10:
            score += 8.0
        elif abs_gap >= 0.07:
            score += 6.0
        elif abs_gap >= 0.05:
            score += 4.0
        else:
            score += abs_gap / 0.05 * 4.0

        # Line movement analysis (0-7 points)
        if line_movement:
            opening = line_movement.get("opening_line")
            current = line_movement.get("current_line")
            if opening is not None and current is not None:
                movement = current - opening

                # Favorable movement (line moved toward our position): +4-7 points
                # We're betting AGAINST the movement = value
                if prob_gap > 0 and movement < 0:
                    # We think higher prob, line dropped = market gives us value
                    score += min(7, abs(movement) * 2)
                elif prob_gap < 0 and movement > 0:
                    # We think lower prob, line rose = market gives us value
                    score += min(7, abs(movement) * 2)
                # Betting WITH sharp movement: moderate confidence
                elif abs(movement) > 1:
                    score += 2.0

        return min(15, score)

    def _score_situational_factors(
        self,
        situational: SituationalFactors,
        bet_type: BetType
    ) -> float:
        """Score based on situational factors (0-10 points)."""
        score = 5.0  # Start at neutral

        # Rest advantage analysis (±2 points)
        rest_diff = situational.home_rest_days - situational.away_rest_days
        if abs(rest_diff) >= 2:
            # Clear rest advantage is more predictable
            score += 1.5
        elif abs(rest_diff) == 0:
            # Both equally rested - neutral
            pass
        else:
            # Slight rest difference - minimal impact
            score += 0.5

        # Back-to-back penalty (both teams tired = less predictable)
        if situational.home_rest_days <= 1 and situational.away_rest_days <= 1:
            score -= 1.5

        # Travel impact (±1 point)
        travel_diff = situational.away_travel_miles - situational.home_travel_miles
        if travel_diff > 500:  # Away team traveled significantly more
            score += 1.0
        elif travel_diff < -500:  # Home team traveled more (unusual)
            score -= 0.5

        # Conference games are more predictable (teams know each other)
        if situational.is_conference_game:
            score += 1.0

        # Rivalry games are LESS predictable
        if situational.is_rivalry_game:
            score -= 1.5

        # Tournament games have different dynamics
        if situational.is_tournament:
            score -= 0.5  # More variance in tournament

        # Season stage affects predictability
        if situational.season_stage == "early":
            score -= 1.5  # Early season = less reliable data
        elif situational.season_stage == "late":
            score += 1.0  # Late season = more reliable
        elif situational.season_stage == "tournament":
            score -= 0.5

        # Recent form divergence
        form_diff = abs(situational.home_recent_form - situational.away_recent_form)
        if form_diff >= 0.4:  # One team much hotter
            score += 1.0
        elif form_diff >= 0.2:
            score += 0.5

        return max(0, min(10, score))

    def _calculate_variance_penalty(
        self,
        home_stats: Optional[TeamStats],
        away_stats: Optional[TeamStats],
        bet_type: BetType,
        stat_comparison: Optional[TeamStatComparison] = None
    ) -> float:
        """Calculate variance penalty for high-uncertainty games (-5 to 0 points).

        Note: Reduced max penalty from -10 to -5 since edge is the primary
        indicator of value. High-edge bets should not be overly penalized.

        NEW: Uses SOS, Pythag, and trend data for more accurate variance assessment.
        """
        penalty = 0.0

        if not home_stats or not away_stats:
            return -1.0  # Slight penalty for missing data

        # High 3-point reliance increases variance (reduced impact)
        avg_3pt = (home_stats.three_point_reliance + away_stats.three_point_reliance) / 2
        if avg_3pt > 0.35:
            penalty -= min(1.5, (avg_3pt - 0.35) * 10)  # Up to -1.5 penalty

        # Low experience increases variance (reduced impact)
        avg_exp = (home_stats.experience + away_stats.experience) / 2
        if avg_exp < 1.5:
            penalty -= min(1.0, (1.5 - avg_exp))  # Up to -1 penalty

        # Tempo mismatch increases unpredictability (reduced)
        if stat_comparison and stat_comparison.tempo_mismatch:
            penalty -= 1.0

        # Both teams inconsistent (high luck ratings) - reduced
        avg_luck = abs(home_stats.luck) + abs(away_stats.luck)
        if avg_luck > 0.06:
            penalty -= 0.5

        # NEW: SOS-based variance adjustment
        # Teams with weak schedules have less reliable ratings
        home_sos_tier = home_stats.schedule_strength_tier
        away_sos_tier = away_stats.schedule_strength_tier
        if home_sos_tier == "weak" or away_sos_tier == "weak":
            penalty -= 0.5  # Weak SOS means less reliable data
        if home_sos_tier == "weak" and away_sos_tier == "weak":
            penalty -= 0.5  # Both weak = even more uncertainty

        # NEW: Trend-based variance
        # Teams in flux (trending up or down significantly) are less predictable
        if abs(home_stats.adj_em_trend) > 3.0 or abs(away_stats.adj_em_trend) > 3.0:
            penalty -= 0.5  # Large rating changes = uncertainty

        # NEW: Pythag reliability bonus (reduces penalty)
        # Teams whose Pythag aligns with record are more predictable
        if home_stats.pythag > 0 and away_stats.pythag > 0:
            # Both teams have reliable Pythag ratings = slight bonus
            if home_stats.rank_pythag > 0 and away_stats.rank_pythag > 0:
                pythag_diff = abs(home_stats.pythag - away_stats.pythag)
                if pythag_diff > 0.4:
                    # Large Pythag gap = more predictable outcome
                    penalty += 0.5  # Reduce penalty (add positive)

        return max(-5, penalty)

    def get_enhanced_confidence(
        self,
        edge: float,
        bet_type: BetType,
        model_prob: float,
        market_implied_prob: float,
        **kwargs
    ) -> Tuple[str, ConfidenceFactors]:
        """
        Get confidence level using enhanced multi-factor scoring.

        Returns both the tier (high/medium/low) and the detailed factors.
        """
        factors = self.calculate_confidence_score(
            edge=edge,
            bet_type=bet_type,
            model_prob=model_prob,
            market_implied_prob=market_implied_prob,
            **kwargs
        )
        return factors.confidence_tier, factors
    
    # ==================== MAIN VALUE FINDING ====================

    def find_value_bets(
        self,
        prediction: GamePrediction,
        market: MarketOdds,
        situational: Optional[SituationalFactors] = None,
        kenpom_prediction: Optional[Dict[str, float]] = None,
        ml_prediction: Optional[Dict[str, float]] = None,
        line_movement: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[ValueBet]:
        """
        Find all value betting opportunities for a game.

        Uses advanced KenPom metrics when available to adjust standard deviations
        and confidence levels for more accurate value detection.

        Args:
            prediction: Our model's prediction (includes stat_comparison if available)
            market: Current market odds
            situational: Optional situational factors (rest, travel, rivalry, etc.)
            kenpom_prediction: Optional KenPom-specific predictions
                {"win_prob": float, "spread": float, "total": float}
            ml_prediction: Optional ML model predictions
                {"win_prob": float, "spread": float, "total": float}
            line_movement: Optional line movement data per bet type
                {"spread": {"opening_line": float, "current_line": float}, ...}

        Returns:
            List of identified value bets
        """
        value_bets = []

        # Extract team stats for advanced calculations if available
        home_stats = None
        away_stats = None
        if prediction.stat_comparison:
            home_stats = prediction.stat_comparison.home_stats
            away_stats = prediction.stat_comparison.away_stats

        # Calculate game-specific standard deviations
        spread_std_dev = self.calculate_adjusted_spread_std_dev(home_stats, away_stats)
        total_std_dev = self.calculate_adjusted_total_std_dev(
            home_stats, away_stats, prediction.tempo
        )

        # Calculate matchup-based edge boost
        edge_boost = self.calculate_matchup_edge_boost(prediction.stat_comparison)

        # Extract model-specific predictions for agreement scoring
        kenpom_win_prob = kenpom_prediction.get("win_prob") if kenpom_prediction else None
        kenpom_total = kenpom_prediction.get("total") if kenpom_prediction else None
        ml_win_prob = ml_prediction.get("win_prob") if ml_prediction else None
        ml_total = ml_prediction.get("total") if ml_prediction else None

        # Check spread bets with adjusted std dev and enhanced confidence
        spread_values = self._check_spread_value(
            prediction, market, spread_std_dev, edge_boost,
            situational=situational,
            kenpom_prob=kenpom_win_prob,
            ml_prob=ml_win_prob,
            line_movement=line_movement.get("spread") if line_movement else None
        )
        value_bets.extend(spread_values)

        # Check total bets with adjusted std dev and enhanced confidence
        total_values = self._check_total_value(
            prediction, market, total_std_dev, edge_boost,
            situational=situational,
            kenpom_total=kenpom_total,
            ml_total=ml_total,
            line_movement=line_movement.get("total") if line_movement else None
        )
        value_bets.extend(total_values)

        # Check moneyline bets with edge boost and enhanced confidence
        ml_values = self._check_moneyline_value(
            prediction, market, edge_boost,
            situational=situational,
            kenpom_win_prob=kenpom_win_prob,
            ml_win_prob=ml_win_prob,
            line_movement=line_movement.get("moneyline") if line_movement else None
        )
        value_bets.extend(ml_values)

        # Sort by confidence score first (if available), then by edge
        value_bets.sort(
            key=lambda x: (x.confidence_score or 0, x.edge),
            reverse=True
        )

        return value_bets

    def _check_spread_value(
        self,
        prediction: GamePrediction,
        market: MarketOdds,
        std_dev: float = None,
        edge_boost: float = 1.0,
        situational: Optional[SituationalFactors] = None,
        kenpom_prob: Optional[float] = None,
        ml_prob: Optional[float] = None,
        line_movement: Optional[Dict[str, float]] = None
    ) -> List[ValueBet]:
        """Check for spread betting value with game-specific adjustments."""
        values = []

        if market.spread == 0:
            return values

        if std_dev is None:
            std_dev = self.BASE_SPREAD_STD_DEV

        # Extract team stats for enhanced confidence scoring
        home_stats = None
        away_stats = None
        stat_comparison = None
        if prediction.stat_comparison:
            stat_comparison = prediction.stat_comparison
            home_stats = prediction.stat_comparison.home_stats
            away_stats = prediction.stat_comparison.away_stats

        # Our predicted margin (positive = home wins by that much)
        predicted_margin = prediction.home_score - prediction.away_score

        # Check home team covering with adjusted std dev
        home_cover_prob = self.cover_spread_prob(predicted_margin, market.spread, std_dev)
        home_implied = self.american_to_implied_prob(market.best_spread_home_odds)
        home_edge = (home_cover_prob - home_implied) * edge_boost

        if home_edge >= self.min_edge_spread:
            # Calculate enhanced confidence
            confidence_tier, confidence_factors = self.get_enhanced_confidence(
                edge=home_edge,
                bet_type=BetType.SPREAD,
                model_prob=home_cover_prob,
                market_implied_prob=home_implied,
                stat_comparison=stat_comparison,
                kenpom_prediction=kenpom_prob,
                ml_prediction=ml_prob,
                situational=situational,
                line_movement=line_movement,
                home_stats=home_stats,
                away_stats=away_stats
            )

            values.append(ValueBet(
                home_team=prediction.home_team,
                away_team=prediction.away_team,
                bet_type=BetType.SPREAD,
                side=prediction.home_team,
                model_prob=home_cover_prob,
                model_line=predicted_margin,
                market_odds=market.best_spread_home_odds,
                market_line=market.spread,
                market_implied_prob=home_implied,
                edge=home_edge,
                ev=self.calculate_ev(home_cover_prob, market.best_spread_home_odds),
                kelly_fraction=self.calculate_kelly(home_cover_prob, market.best_spread_home_odds),
                recommended_book=market.best_spread_home_book,
                confidence=confidence_tier,
                confidence_score=confidence_factors.total_score,
                confidence_factors=confidence_factors.factors_detail
            ))

        # Check away team covering
        away_cover_prob = 1 - home_cover_prob
        away_implied = self.american_to_implied_prob(market.best_spread_away_odds)
        away_edge = (away_cover_prob - away_implied) * edge_boost

        if away_edge >= self.min_edge_spread:
            # Calculate enhanced confidence
            confidence_tier, confidence_factors = self.get_enhanced_confidence(
                edge=away_edge,
                bet_type=BetType.SPREAD,
                model_prob=away_cover_prob,
                market_implied_prob=away_implied,
                stat_comparison=stat_comparison,
                kenpom_prediction=1 - kenpom_prob if kenpom_prob else None,
                ml_prediction=1 - ml_prob if ml_prob else None,
                situational=situational,
                line_movement=line_movement,
                home_stats=home_stats,
                away_stats=away_stats
            )

            values.append(ValueBet(
                home_team=prediction.home_team,
                away_team=prediction.away_team,
                bet_type=BetType.SPREAD,
                side=prediction.away_team,
                model_prob=away_cover_prob,
                model_line=-predicted_margin,
                market_odds=market.best_spread_away_odds,
                market_line=-market.spread,
                market_implied_prob=away_implied,
                edge=away_edge,
                ev=self.calculate_ev(away_cover_prob, market.best_spread_away_odds),
                kelly_fraction=self.calculate_kelly(away_cover_prob, market.best_spread_away_odds),
                recommended_book=market.best_spread_away_book,
                confidence=confidence_tier,
                confidence_score=confidence_factors.total_score,
                confidence_factors=confidence_factors.factors_detail
            ))

        return values

    def _check_total_value(
        self,
        prediction: GamePrediction,
        market: MarketOdds,
        std_dev: float = 12.0,
        edge_boost: float = 1.0,
        situational: Optional[SituationalFactors] = None,
        kenpom_total: Optional[float] = None,
        ml_total: Optional[float] = None,
        line_movement: Optional[Dict[str, float]] = None
    ) -> List[ValueBet]:
        """Check for over/under betting value with game-specific adjustments."""
        values = []

        if market.total == 0:
            return values

        # Extract team stats for enhanced confidence scoring
        home_stats = None
        away_stats = None
        stat_comparison = None
        if prediction.stat_comparison:
            stat_comparison = prediction.stat_comparison
            home_stats = prediction.stat_comparison.home_stats
            away_stats = prediction.stat_comparison.away_stats

        over_prob, under_prob = self.over_under_prob(prediction.total, market.total, std_dev)

        # Calculate model agreement for totals (convert to pseudo-probability)
        kenpom_over_prob = None
        ml_over_prob = None
        if kenpom_total is not None:
            kenpom_over_prob = self.over_under_prob(kenpom_total, market.total, std_dev)[0]
        if ml_total is not None:
            ml_over_prob = self.over_under_prob(ml_total, market.total, std_dev)[0]

        # Check over with edge boost
        over_implied = self.american_to_implied_prob(market.best_over_odds)
        over_edge = (over_prob - over_implied) * edge_boost

        if over_edge >= self.min_edge_total:
            # Calculate enhanced confidence
            confidence_tier, confidence_factors = self.get_enhanced_confidence(
                edge=over_edge,
                bet_type=BetType.TOTAL,
                model_prob=over_prob,
                market_implied_prob=over_implied,
                stat_comparison=stat_comparison,
                kenpom_prediction=kenpom_over_prob,
                ml_prediction=ml_over_prob,
                situational=situational,
                line_movement=line_movement,
                home_stats=home_stats,
                away_stats=away_stats
            )

            values.append(ValueBet(
                home_team=prediction.home_team,
                away_team=prediction.away_team,
                bet_type=BetType.TOTAL,
                side="over",
                model_prob=over_prob,
                model_line=prediction.total,
                market_odds=market.best_over_odds,
                market_line=market.total,
                market_implied_prob=over_implied,
                edge=over_edge,
                ev=self.calculate_ev(over_prob, market.best_over_odds),
                kelly_fraction=self.calculate_kelly(over_prob, market.best_over_odds),
                recommended_book=market.best_over_book,
                confidence=confidence_tier,
                confidence_score=confidence_factors.total_score,
                confidence_factors=confidence_factors.factors_detail
            ))

        # Check under with edge boost
        under_implied = self.american_to_implied_prob(market.best_under_odds)
        under_edge = (under_prob - under_implied) * edge_boost

        if under_edge >= self.min_edge_total:
            # Calculate enhanced confidence
            confidence_tier, confidence_factors = self.get_enhanced_confidence(
                edge=under_edge,
                bet_type=BetType.TOTAL,
                model_prob=under_prob,
                market_implied_prob=under_implied,
                stat_comparison=stat_comparison,
                kenpom_prediction=1 - kenpom_over_prob if kenpom_over_prob else None,
                ml_prediction=1 - ml_over_prob if ml_over_prob else None,
                situational=situational,
                line_movement=line_movement,
                home_stats=home_stats,
                away_stats=away_stats
            )

            values.append(ValueBet(
                home_team=prediction.home_team,
                away_team=prediction.away_team,
                bet_type=BetType.TOTAL,
                side="under",
                model_prob=under_prob,
                model_line=prediction.total,
                market_odds=market.best_under_odds,
                market_line=market.total,
                market_implied_prob=under_implied,
                edge=under_edge,
                ev=self.calculate_ev(under_prob, market.best_under_odds),
                kelly_fraction=self.calculate_kelly(under_prob, market.best_under_odds),
                recommended_book=market.best_under_book,
                confidence=confidence_tier,
                confidence_score=confidence_factors.total_score,
                confidence_factors=confidence_factors.factors_detail
            ))

        return values

    def _check_moneyline_value(
        self,
        prediction: GamePrediction,
        market: MarketOdds,
        edge_boost: float = 1.0,
        situational: Optional[SituationalFactors] = None,
        kenpom_win_prob: Optional[float] = None,
        ml_win_prob: Optional[float] = None,
        line_movement: Optional[Dict[str, float]] = None
    ) -> List[ValueBet]:
        """Check for moneyline betting value with matchup adjustment."""
        values = []

        # Need valid moneyline odds to calculate value
        if market.best_ml_home_odds == 0 or market.best_ml_away_odds == 0:
            return values

        # Extract team stats for enhanced confidence scoring
        home_stats = None
        away_stats = None
        stat_comparison = None
        if prediction.stat_comparison:
            stat_comparison = prediction.stat_comparison
            home_stats = prediction.stat_comparison.home_stats
            away_stats = prediction.stat_comparison.away_stats

        # Check home moneyline with edge boost
        home_implied = self.american_to_implied_prob(market.best_ml_home_odds)
        home_edge = (prediction.home_win_prob - home_implied) * edge_boost

        if home_edge >= self.min_edge_ml:
            # Calculate enhanced confidence
            confidence_tier, confidence_factors = self.get_enhanced_confidence(
                edge=home_edge,
                bet_type=BetType.MONEYLINE,
                model_prob=prediction.home_win_prob,
                market_implied_prob=home_implied,
                stat_comparison=stat_comparison,
                kenpom_prediction=kenpom_win_prob,
                ml_prediction=ml_win_prob,
                situational=situational,
                line_movement=line_movement,
                home_stats=home_stats,
                away_stats=away_stats
            )

            values.append(ValueBet(
                home_team=prediction.home_team,
                away_team=prediction.away_team,
                bet_type=BetType.MONEYLINE,
                side=prediction.home_team,
                model_prob=prediction.home_win_prob,
                model_line=None,
                market_odds=market.best_ml_home_odds,
                market_line=None,
                market_implied_prob=home_implied,
                edge=home_edge,
                ev=self.calculate_ev(prediction.home_win_prob, market.best_ml_home_odds),
                kelly_fraction=self.calculate_kelly(prediction.home_win_prob, market.best_ml_home_odds),
                recommended_book=market.best_ml_home_book,
                confidence=confidence_tier,
                confidence_score=confidence_factors.total_score,
                confidence_factors=confidence_factors.factors_detail
            ))

        # Check away moneyline with edge boost
        away_implied = self.american_to_implied_prob(market.best_ml_away_odds)
        away_edge = (prediction.away_win_prob - away_implied) * edge_boost

        if away_edge >= self.min_edge_ml:
            # Calculate enhanced confidence
            confidence_tier, confidence_factors = self.get_enhanced_confidence(
                edge=away_edge,
                bet_type=BetType.MONEYLINE,
                model_prob=prediction.away_win_prob,
                market_implied_prob=away_implied,
                stat_comparison=stat_comparison,
                kenpom_prediction=1 - kenpom_win_prob if kenpom_win_prob else None,
                ml_prediction=1 - ml_win_prob if ml_win_prob else None,
                situational=situational,
                line_movement=line_movement,
                home_stats=home_stats,
                away_stats=away_stats
            )

            values.append(ValueBet(
                home_team=prediction.home_team,
                away_team=prediction.away_team,
                bet_type=BetType.MONEYLINE,
                side=prediction.away_team,
                model_prob=prediction.away_win_prob,
                model_line=None,
                market_odds=market.best_ml_away_odds,
                market_line=None,
                market_implied_prob=away_implied,
                edge=away_edge,
                ev=self.calculate_ev(prediction.away_win_prob, market.best_ml_away_odds),
                kelly_fraction=self.calculate_kelly(prediction.away_win_prob, market.best_ml_away_odds),
                recommended_book=market.best_ml_away_book,
                confidence=confidence_tier,
                confidence_score=confidence_factors.total_score,
                confidence_factors=confidence_factors.factors_detail
            ))

        return values
    
    # ==================== FORMATTING ====================
    
    def format_value_bet(self, vb: ValueBet, show_details: bool = False) -> str:
        """Format a value bet for display."""
        # Build confidence display
        if vb.confidence_score is not None:
            conf_str = f"{vb.confidence.upper()} ({vb.confidence_score:.0f}/100)"
        else:
            conf_str = vb.confidence.upper()

        if vb.bet_type == BetType.SPREAD:
            line_str = f"{vb.market_line:+.1f}" if vb.market_line else ""
            result = (
                f"📊 SPREAD: {vb.side} {line_str} ({vb.market_odds:+d})\n"
                f"   Model: {vb.model_prob:.1%} | Market: {vb.market_implied_prob:.1%}\n"
                f"   Edge: {vb.edge:.1%} | EV: {vb.ev:+.1%} | Kelly: {vb.kelly_fraction:.1%}\n"
                f"   Best Book: {vb.recommended_book} | Confidence: {conf_str}"
            )
        elif vb.bet_type == BetType.TOTAL:
            result = (
                f"📈 TOTAL: {vb.side.upper()} {vb.market_line} ({vb.market_odds:+d})\n"
                f"   Model Total: {vb.model_line:.1f} | Line: {vb.market_line}\n"
                f"   Model: {vb.model_prob:.1%} | Market: {vb.market_implied_prob:.1%}\n"
                f"   Edge: {vb.edge:.1%} | EV: {vb.ev:+.1%} | Kelly: {vb.kelly_fraction:.1%}\n"
                f"   Best Book: {vb.recommended_book} | Confidence: {conf_str}"
            )
        else:  # MONEYLINE
            result = (
                f"💰 MONEYLINE: {vb.side} ({vb.market_odds:+d})\n"
                f"   Model Win%: {vb.model_prob:.1%} | Market: {vb.market_implied_prob:.1%}\n"
                f"   Edge: {vb.edge:.1%} | EV: {vb.ev:+.1%} | Kelly: {vb.kelly_fraction:.1%}\n"
                f"   Best Book: {vb.recommended_book} | Confidence: {conf_str}"
            )

        # Add detailed breakdown if requested and available
        if show_details and vb.confidence_factors:
            result += f"\n   --- Confidence Breakdown ---"
            factors = vb.confidence_factors

            if "edge" in factors:
                result += f"\n   Edge: {factors['edge'].get('interpretation', 'N/A')} ({factors['edge'].get('score', 0):.0f} pts)"

            if "statistical" in factors:
                stat = factors["statistical"]
                edges = []
                if stat.get("efficiency_edge"):
                    edges.append(f"efficiency ({stat['efficiency_edge']})")
                if stat.get("shooting_edge"):
                    edges.append(f"shooting ({stat['shooting_edge']})")
                if stat.get("rebounding_edge"):
                    edges.append(f"rebounding ({stat['rebounding_edge']})")
                if stat.get("turnover_edge"):
                    edges.append(f"turnovers ({stat['turnover_edge']})")
                if edges:
                    result += f"\n   Statistical: {', '.join(edges)} ({stat.get('score', 0):.0f} pts)"
                else:
                    result += f"\n   Statistical: No major edges ({stat.get('score', 0):.0f} pts)"

            if "model_agreement" in factors:
                ma = factors["model_agreement"]
                if ma.get("score", 0) >= 12:
                    result += f"\n   Models: Strong agreement ({ma.get('score', 0):.0f} pts)"
                elif ma.get("score", 0) >= 7:
                    result += f"\n   Models: Moderate agreement ({ma.get('score', 0):.0f} pts)"
                else:
                    result += f"\n   Models: Weak/single source ({ma.get('score', 0):.0f} pts)"

            if "variance" in factors:
                penalty = factors["variance"].get("penalty", 0)
                if penalty <= -5:
                    result += f"\n   ⚠️  High-variance matchup ({penalty:.0f} pts)"
                elif penalty < 0:
                    result += f"\n   Variance: Moderate ({penalty:.0f} pts)"

        return result

    # ==================== STAT COMPARISON ====================

    @staticmethod
    def compare_team_stats(
        home_stats: TeamStats,
        away_stats: TeamStats
    ) -> TeamStatComparison:
        """
        Compare statistics between two teams and identify major differences.

        Args:
            home_stats: Home team's KenPom stats
            away_stats: Away team's KenPom stats

        Returns:
            TeamStatComparison with all differences and insights
        """
        differences = []

        # Define stat comparisons: (stat_name, display_name, threshold_major, threshold_moderate, higher_is_better)
        stat_configs = [
            ("adj_em", "Efficiency Margin", 8.0, 4.0, True),
            ("adj_oe", "Offensive Efficiency", 5.0, 2.5, True),
            ("adj_de", "Defensive Efficiency", 5.0, 2.5, False),  # Lower is better for defense
            ("adj_tempo", "Tempo", 5.0, 2.5, None),  # Neutral - just shows mismatch
            ("efg_pct", "Effective FG%", 4.0, 2.0, True),
            ("to_pct", "Turnover %", 3.0, 1.5, False),  # Lower is better
            ("or_pct", "Off. Rebound %", 4.0, 2.0, True),
            ("ft_rate", "FT Rate", 6.0, 3.0, True),
            ("d_efg_pct", "Opp. eFG%", 4.0, 2.0, False),  # Lower is better (better D)
            ("d_to_pct", "Forced TO %", 3.0, 1.5, True),  # Higher is better (forces more)
            ("d_or_pct", "Def. Rebound %", 4.0, 2.0, False),  # Lower means better DRB
            ("d_ft_rate", "Opp. FT Rate", 6.0, 3.0, False),  # Lower is better
            ("sos", "Strength of Sched", 3.0, 1.5, True),
        ]

        tempo_mismatch = False
        efficiency_edge = ""
        shooting_edge = ""
        rebounding_edge = ""
        turnover_edge = ""

        for stat_name, display_name, major_thresh, moderate_thresh, higher_is_better in stat_configs:
            home_val = getattr(home_stats, stat_name, 0.0)
            away_val = getattr(away_stats, stat_name, 0.0)
            diff = home_val - away_val

            # Determine significance
            abs_diff = abs(diff)
            if abs_diff >= major_thresh:
                significance = "major"
            elif abs_diff >= moderate_thresh:
                significance = "moderate"
            else:
                significance = "minor"

            # Determine advantage
            if higher_is_better is None:  # Tempo - just a mismatch indicator
                advantage = "neutral"
                if abs_diff >= major_thresh:
                    tempo_mismatch = True
            elif higher_is_better:
                advantage = "home" if diff > 0 else "away" if diff < 0 else "neutral"
            else:
                advantage = "home" if diff < 0 else "away" if diff > 0 else "neutral"

            differences.append(StatDifference(
                stat_name=stat_name,
                display_name=display_name,
                home_value=home_val,
                away_value=away_val,
                difference=diff,
                advantage=advantage,
                significance=significance,
                higher_is_better=higher_is_better if higher_is_better is not None else True
            ))

            # Track summary edges for major differences
            if significance == "major":
                if stat_name == "adj_em":
                    efficiency_edge = "home" if diff > 0 else "away"
                elif stat_name in ("efg_pct", "d_efg_pct"):
                    if not shooting_edge:
                        shooting_edge = advantage
                elif stat_name in ("or_pct", "d_or_pct"):
                    if not rebounding_edge:
                        rebounding_edge = advantage
                elif stat_name in ("to_pct", "d_to_pct"):
                    if not turnover_edge:
                        turnover_edge = advantage

        return TeamStatComparison(
            home_stats=home_stats,
            away_stats=away_stats,
            stat_differences=differences,
            efficiency_edge=efficiency_edge,
            tempo_mismatch=tempo_mismatch,
            shooting_edge=shooting_edge,
            rebounding_edge=rebounding_edge,
            turnover_edge=turnover_edge
        )
