"""
Game Service

Combines KenPom predictions with betting odds to provide
game analysis and value bet identification.
"""

import asyncio
import logging
import threading
from datetime import date, datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from clients import KenPomClient, OddsAPIClient, Market
from .value_calculator import (
    ValueCalculator,
    GamePrediction,
    MarketOdds,
    ValueBet,
    TeamStats,
    TeamStatComparison,
    StatDifference
)

logger = logging.getLogger(__name__)


@dataclass
class GameAnalysis:
    """Complete analysis for a single game."""
    # Basic info
    home_team: str
    away_team: str
    game_time: datetime

    # KenPom predictions
    kenpom_home_score: float
    kenpom_away_score: float
    kenpom_spread: float
    kenpom_total: float
    kenpom_home_win_prob: float
    kenpom_tempo: float

    # Vegas lines
    vegas_spread: float
    vegas_total: float
    vegas_ml_home: int
    vegas_ml_away: int

    # Spread comparison
    spread_diff: float  # KenPom spread - Vegas spread
    total_diff: float   # KenPom total - Vegas total

    # Value bets
    value_bets: List[ValueBet]

    # Rankings
    home_rank: int = 0
    away_rank: int = 0

    # Team statistics comparison
    stat_comparison: Optional[TeamStatComparison] = None

    # Summary of major differences
    major_stat_diffs: List[StatDifference] = field(default_factory=list)


class GameService:
    """
    Service for fetching and analyzing games.
    """

    def __init__(
        self,
        kenpom_client: Optional[KenPomClient] = None,
        odds_client: Optional[OddsAPIClient] = None,
        value_calculator: Optional[ValueCalculator] = None
    ):
        self.kenpom = kenpom_client or KenPomClient()
        self.odds = odds_client or OddsAPIClient()
        self.calculator = value_calculator or ValueCalculator()

        # Cache for team name mapping
        self._team_name_cache: Dict[str, str] = {}

        # Cache for team stats (ratings + four factors + extended data)
        self._ratings_cache: Dict[str, Dict[str, Any]] = {}
        self._four_factors_cache: Dict[str, Dict[str, Any]] = {}
        self._point_dist_cache: Dict[str, Dict[str, Any]] = {}
        self._height_cache: Dict[str, Dict[str, Any]] = {}
        self._misc_stats_cache: Dict[str, Dict[str, Any]] = {}
        self._archive_cache: Dict[str, Dict[str, Any]] = {}  # Historical ratings for trends
        self._cache_loaded: bool = False

        # Thread lock for cache access
        self._cache_lock = threading.RLock()
    
    async def _load_team_stats_cache(self) -> None:
        """Load all KenPom data into cache for comprehensive analysis."""
        with self._cache_lock:
            if self._cache_loaded:
                return

        try:
            # Calculate date for historical comparison (14 days ago)
            from datetime import timedelta
            archive_date = (date.today() - timedelta(days=14)).isoformat()

            # Fetch all available KenPom data concurrently
            results = await asyncio.gather(
                self.kenpom.get_ratings(),
                self.kenpom.get_four_factors(),
                self.kenpom.get_point_distribution(),
                self.kenpom.get_height(),
                self.kenpom.get_misc_stats(),
                self.kenpom.get_archive(date_str=archive_date),  # Historical data for trends
                return_exceptions=True  # Don't fail if one endpoint fails
            )

            ratings, four_factors, point_dist, height, misc_stats, archive = results

            # Build lookup dictionaries by team name
            def build_cache(data, cache_dict):
                if isinstance(data, Exception):
                    logger.debug(f"Cache build skipped due to exception: {data}")
                    return
                if not data:
                    return
                for item in data:
                    team_name = item.get("TeamName", "").lower()
                    if team_name:
                        cache_dict[team_name] = item

            with self._cache_lock:
                build_cache(ratings, self._ratings_cache)
                build_cache(four_factors, self._four_factors_cache)
                build_cache(point_dist, self._point_dist_cache)
                build_cache(height, self._height_cache)
                build_cache(misc_stats, self._misc_stats_cache)
                build_cache(archive, self._archive_cache)
                self._cache_loaded = True

            logger.info(f"Loaded KenPom cache: {len(self._ratings_cache)} teams, "
                       f"{len(self._archive_cache)} archive entries")

        except Exception as e:
            # Log but don't fail - stats are optional enhancement
            logger.warning(f"Could not load team stats cache: {e}")

    def _get_team_stats(self, team_name: str) -> TeamStats:
        """Get TeamStats for a team from cache with all available KenPom data."""
        normalized = team_name.lower()

        def find_in_cache(cache: Dict, name: str) -> Dict:
            """Find team in cache, trying exact match then partial match."""
            # Try exact match first
            result = cache.get(name, {})
            if result:
                return result
            # Try partial match
            for cached_name, data in cache.items():
                if name in cached_name or cached_name in name:
                    return data
            return {}

        # Get data from all caches (thread-safe)
        with self._cache_lock:
            ratings = find_in_cache(self._ratings_cache, normalized)
            four_factors = find_in_cache(self._four_factors_cache, normalized)
            point_dist = find_in_cache(self._point_dist_cache, normalized)
            height = find_in_cache(self._height_cache, normalized)
            misc_stats = find_in_cache(self._misc_stats_cache, normalized)
            archive = find_in_cache(self._archive_cache, normalized)

        # Calculate trends (current - historical)
        adj_em_trend = 0.0
        adj_oe_trend = 0.0
        adj_de_trend = 0.0
        if ratings and archive:
            current_em = ratings.get("AdjEM", 0.0)
            archive_em = archive.get("AdjEM", 0.0)
            if current_em and archive_em:
                adj_em_trend = current_em - archive_em

            current_oe = ratings.get("AdjOE", 0.0)
            archive_oe = archive.get("AdjOE", 0.0)
            if current_oe and archive_oe:
                adj_oe_trend = current_oe - archive_oe

            current_de = ratings.get("AdjDE", 0.0)
            archive_de = archive.get("AdjDE", 0.0)
            if current_de and archive_de:
                # For defense, lower is better, so positive trend = improvement
                adj_de_trend = archive_de - current_de

        return TeamStats(
            team_name=team_name,
            # Efficiency ratings
            rank=ratings.get("RankAdjEM", 0),
            adj_em=ratings.get("AdjEM", 0.0),
            adj_oe=ratings.get("AdjOE", 0.0),
            adj_de=ratings.get("AdjDE", 0.0),
            adj_tempo=ratings.get("AdjTempo", 0.0),
            sos=ratings.get("SOS", 0.0),
            luck=ratings.get("Luck", 0.0),

            # NEW: Pythagorean rating (better win probability predictor)
            pythag=ratings.get("Pythag", 0.0),
            rank_pythag=ratings.get("RankPythag", 0),

            # NEW: Average Possession Length (for totals predictions)
            apl_off=ratings.get("APL_Off", 0.0),
            apl_def=ratings.get("APL_Def", 0.0),
            rank_apl_off=ratings.get("RankAPL_Off", 0),
            rank_apl_def=ratings.get("RankAPL_Def", 0),

            # NEW: Conference-specific APL
            conf_apl_off=ratings.get("ConfAPL_Off", 0.0),
            conf_apl_def=ratings.get("ConfAPL_Def", 0.0),

            # NEW: Strength of Schedule Components
            sos_off=ratings.get("SOSO", 0.0),
            sos_def=ratings.get("SOSD", 0.0),
            rank_sos=ratings.get("RankSOS", 0),
            rank_sos_off=ratings.get("RankSOSO", 0),
            rank_sos_def=ratings.get("RankSOSD", 0),
            nc_sos=ratings.get("NCSOS", 0.0),
            rank_nc_sos=ratings.get("RankNCSOS", 0),

            # NEW: Conference info
            conference=ratings.get("ConfShort", ""),

            # Four Factors
            efg_pct=four_factors.get("eFG_Pct", 0.0),
            to_pct=four_factors.get("TO_Pct", 0.0),
            or_pct=four_factors.get("OR_Pct", 0.0),
            ft_rate=four_factors.get("FT_Rate", 0.0),
            d_efg_pct=four_factors.get("DeFG_Pct", 0.0),
            d_to_pct=four_factors.get("DTO_Pct", 0.0),
            d_or_pct=four_factors.get("DOR_Pct", 0.0),
            d_ft_rate=four_factors.get("DFT_Rate", 0.0),

            # Point Distribution
            off_ft_pct=point_dist.get("FTPct", 0.0),
            off_2pt_pct=point_dist.get("TwoPtPct", 0.0),
            off_3pt_pct=point_dist.get("ThreePtPct", 0.0),
            def_ft_pct=point_dist.get("DefFTPct", 0.0),
            def_2pt_pct=point_dist.get("DefTwoPtPct", 0.0),
            def_3pt_pct=point_dist.get("DefThreePtPct", 0.0),
            # Height & Experience
            avg_height=height.get("AvgHgt", 0.0),
            experience=height.get("Experience", 0.0),
            bench_minutes=height.get("Bench", 0.0),
            continuity=height.get("Continuity", 0.0),
            # Misc Stats
            fg_pct_2pt=misc_stats.get("FG2Pct", 0.0),
            fg_pct_3pt=misc_stats.get("FG3Pct", 0.0),
            ft_pct=misc_stats.get("FTPct", 0.0),
            blk_pct=misc_stats.get("BlockPct", 0.0),
            stl_pct=misc_stats.get("StlRate", 0.0),
            ast_ratio=misc_stats.get("AstRate", 0.0),
            opp_fg_pct_2pt=misc_stats.get("OppFG2Pct", 0.0),
            opp_fg_pct_3pt=misc_stats.get("OppFG3Pct", 0.0),

            # NEW: Trend data (rating change over last 2 weeks)
            adj_em_trend=adj_em_trend,
            adj_oe_trend=adj_oe_trend,
            adj_de_trend=adj_de_trend,
        )

    async def get_todays_analysis(self) -> List[GameAnalysis]:
        """
        Get complete analysis for all of today's games.

        Returns:
            List of GameAnalysis objects
        """
        return await self.get_analysis_for_date(date.today().isoformat())

    async def get_analysis_for_date(self, game_date: str) -> List[GameAnalysis]:
        """
        Get complete analysis for all games on a specific date.

        Args:
            game_date: Date in YYYY-MM-DD format

        Returns:
            List of GameAnalysis objects
        """
        # Fetch data concurrently (including team stats for comparisons)
        kenpom_games, odds_games, _ = await asyncio.gather(
            self.kenpom.get_fanmatch(game_date),
            self.odds.get_all_odds(),
            self._load_team_stats_cache()
        )

        # Match games and analyze
        analyses = []
        for kp_game in kenpom_games:
            # Find matching odds game
            odds_game = self._match_odds_game(kp_game, odds_games)

            if odds_game:
                analysis = self._create_analysis(kp_game, odds_game)
                analyses.append(analysis)
            else:
                # Create analysis with KenPom only (no odds available)
                analysis = self._create_analysis_kenpom_only(kp_game)
                analyses.append(analysis)

        # Sort by game time
        analyses.sort(key=lambda x: x.game_time)

        return analyses
    
    async def get_value_bets_today(self, min_edge: float = 0.03) -> List[Tuple[GameAnalysis, List[ValueBet]]]:
        """
        Get all value bets for today's games.
        
        Args:
            min_edge: Minimum edge to include
            
        Returns:
            List of (game, value_bets) tuples
        """
        analyses = await self.get_todays_analysis()
        
        value_games = []
        for analysis in analyses:
            if analysis.value_bets:
                # Filter by minimum edge
                filtered = [vb for vb in analysis.value_bets if vb.edge >= min_edge]
                if filtered:
                    value_games.append((analysis, filtered))
        
        # Sort by best edge
        value_games.sort(
            key=lambda x: max(vb.edge for vb in x[1]),
            reverse=True
        )
        
        return value_games
    
    async def get_game_analysis(
        self,
        home_team: str,
        away_team: str,
        game_date: str = None
    ) -> Optional[GameAnalysis]:
        """
        Get analysis for a specific matchup.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Date in YYYY-MM-DD format (defaults to today)
            
        Returns:
            GameAnalysis or None if not found
        """
        if game_date is None:
            game_date = date.today().isoformat()
        
        kenpom_games = await self.kenpom.get_fanmatch(game_date)
        odds_games = await self.odds.get_all_odds()
        
        # Find matching KenPom game
        kp_game = None
        for game in kenpom_games:
            if (self._normalize_name(game["Home"]) == self._normalize_name(home_team) and
                self._normalize_name(game["Visitor"]) == self._normalize_name(away_team)):
                kp_game = game
                break
        
        if not kp_game:
            return None
        
        odds_game = self._match_odds_game(kp_game, odds_games)
        
        if odds_game:
            return self._create_analysis(kp_game, odds_game)
        else:
            return self._create_analysis_kenpom_only(kp_game)
    
    # ==================== MATCHING HELPERS ====================
    
    def _match_odds_game(
        self,
        kp_game: Dict[str, Any],
        odds_games: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Match a KenPom game to an Odds API game."""
        kp_home = self._normalize_name(kp_game["Home"])
        kp_away = self._normalize_name(kp_game["Visitor"])
        
        for odds_game in odds_games:
            odds_home = self._normalize_name(odds_game["home_team"])
            odds_away = self._normalize_name(odds_game["away_team"])
            
            # Try direct match
            if kp_home == odds_home and kp_away == odds_away:
                return odds_game
            
            # Try fuzzy match
            if (self._fuzzy_match(kp_home, odds_home) and 
                self._fuzzy_match(kp_away, odds_away)):
                return odds_game
        
        return None
    
    # Common team name mappings for difficult matches
    TEAM_NAME_ALIASES = {
        # Saint variations
        "saint mary's": "st. mary's",
        "saint marys": "st. mary's",
        "st marys": "st. mary's",
        "st mary's": "st. mary's",
        "saint joseph's": "st. joseph's",
        "saint josephs": "st. joseph's",
        "saint john's": "st. john's",
        "saint johns": "st. john's",
        "saint louis": "st. louis",
        "saint bonaventure": "st. bonaventure",
        "saint peter's": "st. peter's",

        # State abbreviations
        "nc state": "north carolina state",
        "n.c. state": "north carolina state",

        # UC schools
        "uc irvine": "irvine",
        "uc davis": "davis",
        "uc riverside": "riverside",
        "uc santa barbara": "santa barbara",
        "uc san diego": "san diego",
        "ucsb": "santa barbara",
        "ucsd": "san diego",
        "uci": "irvine",

        # Miami disambiguation
        "miami (oh)": "miami oh",
        "miami (fl)": "miami fl",
        "miami ohio": "miami oh",
        "miami florida": "miami fl",

        # USC/UCLA common names
        "southern california": "usc",
        "southern cal": "usc",

        # Common abbreviations
        "louisiana state": "lsu",
        "brigham young": "byu",
        "texas christian": "tcu",
        "southern methodist": "smu",
        "virginia commonwealth": "vcu",
        "central florida": "ucf",
        "connecticut": "uconn",
        "massachusetts": "umass",

        # Other common variations
        "ole miss": "mississippi",
        "pitt": "pittsburgh",
        "cal": "california",
        "penn": "pennsylvania",
    }

    # Common mascot suffixes to remove
    MASCOT_SUFFIXES = [
        " blue devils", " wildcats", " bulldogs", " tigers", " bears",
        " crimson tide", " fighting irish", " volunteers", " gators",
        " demon deacons", " tar heels", " hokies", " cavaliers",
        " jayhawks", " sooners", " longhorns", " aggies", " bruins",
        " trojans", " ducks", " huskies", " cougars", " spartans",
        " wolverines", " buckeyes", " nittany lions", " hoosiers",
        " boilermakers", " hawkeyes", " badgers", " golden gophers",
        " cornhuskers", " cyclones", " mountaineers", " red raiders",
        " horned frogs", " razorbacks", " rebels", " commodores",
        " gamecocks", " seminoles", " hurricanes", " yellow jackets",
        " cardinals", " blue jays", " orange", " golden eagles",
        " musketeers", " bearcats", " pirates", " red storm",
        " friars", " hoyas", " bluejays", " creighton bluejays"
    ]

    def _normalize_name(self, name: str) -> str:
        """Normalize team name for matching."""
        if not name:
            return ""

        name = name.lower().strip()

        # Remove mascot suffixes
        for suffix in self.MASCOT_SUFFIXES:
            if name.endswith(suffix):
                name = name[:-len(suffix)]

        # Standardize common patterns
        name = name.replace(" st.", " state").replace(" st ", " state ")
        name = name.replace("'", "").replace("'", "")  # Remove apostrophes
        name = name.replace("-", " ")  # Replace hyphens with spaces

        # Check for known aliases
        if name in self.TEAM_NAME_ALIASES:
            name = self.TEAM_NAME_ALIASES[name]

        # Remove "university" and "college" suffixes
        name = name.replace(" university", "").replace(" college", "")

        return name.strip()
    
    def _fuzzy_match(self, name1: str, name2: str) -> bool:
        """Check if two team names likely refer to the same team."""
        # Check if one is contained in the other
        if name1 in name2 or name2 in name1:
            return True

        # Check for common words
        words1 = set(name1.split())
        words2 = set(name2.split())
        common = words1 & words2

        # If they share at least one meaningful word
        stopwords = {"state", "university", "college", "of", "the", "a", "and"}
        meaningful = common - stopwords

        # Require the first word to match for better accuracy
        # This prevents "Kansas" from matching "Kansas State"
        first_word1 = name1.split()[0] if name1.split() else ""
        first_word2 = name2.split()[0] if name2.split() else ""

        if len(meaningful) >= 1 and first_word1 == first_word2:
            return True

        # Calculate similarity ratio - use stricter threshold (0.9)
        similarity = self._similarity_ratio(name1, name2)
        return similarity >= 0.9

    @staticmethod
    def _similarity_ratio(s1: str, s2: str) -> float:
        """
        Calculate a simple similarity ratio between two strings.
        Returns a value between 0.0 (no match) and 1.0 (exact match).
        """
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        # Use simple character-based Jaccard similarity
        set1 = set(s1.replace(" ", ""))
        set2 = set(s2.replace(" ", ""))
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _normalize_win_probability(prob: float) -> float:
        """
        Normalize win probability to decimal form (0.0 to 1.0).

        KenPom API may return probabilities as:
        - Decimal (0.95) - already correct
        - Percentage (95) - needs division by 100
        - Sometimes as string - needs conversion

        Args:
            prob: Raw probability value from API

        Returns:
            Probability as decimal between 0.0 and 1.0
        """
        # Handle string input
        if isinstance(prob, str):
            try:
                prob = float(prob)
            except (ValueError, TypeError):
                return 0.5  # Default to 50% if unparseable

        # Handle None or invalid values
        if prob is None or not isinstance(prob, (int, float)):
            return 0.5

        # If greater than 1, assume it's a percentage (0-100 scale)
        if prob > 1:
            prob = prob / 100.0

        # Clamp to valid probability range
        return max(0.0, min(1.0, prob))
    
    # ==================== ANALYSIS CREATION ====================
    
    def _create_analysis(
        self,
        kp_game: Dict[str, Any],
        odds_game: Dict[str, Any]
    ) -> GameAnalysis:
        """Create a complete game analysis."""
        # Parse odds
        market_odds = self._parse_odds(odds_game)

        # Create prediction object
        # KenPom can return win probability as either decimal (0.95) or percentage (95)
        # Handle both cases robustly
        home_win_prob = self._normalize_win_probability(kp_game.get("HomeWP", 0.5))

        # Get team stats and create comparison
        home_stats = self._get_team_stats(kp_game["Home"])
        away_stats = self._get_team_stats(kp_game["Visitor"])
        stat_comparison = self.calculator.compare_team_stats(home_stats, away_stats)

        prediction = GamePrediction(
            home_team=kp_game["Home"],
            away_team=kp_game["Visitor"],
            home_score=kp_game["HomePred"],
            away_score=kp_game["VisitorPred"],
            home_win_prob=home_win_prob,
            tempo=kp_game.get("PredTempo", 0),
            stat_comparison=stat_comparison
        )

        # Find value bets
        value_bets = self.calculator.find_value_bets(prediction, market_odds)

        # KenPom spread (negative = home favored, matching betting convention)
        kp_spread = kp_game["VisitorPred"] - kp_game["HomePred"]
        kp_total = kp_game["HomePred"] + kp_game["VisitorPred"]

        # Safely parse game time
        try:
            game_time = datetime.fromisoformat(odds_game["commence_time"].replace("Z", "+00:00"))
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Could not parse game time: {e}, using current time")
            game_time = datetime.now()

        # Safely calculate spread/total diffs (handle None values)
        vegas_spread = market_odds.spread if market_odds.spread is not None else 0.0
        vegas_total = market_odds.total if market_odds.total is not None else 0.0
        spread_diff = kp_spread - vegas_spread if vegas_spread != 0 else 0.0
        total_diff = kp_total - vegas_total if vegas_total != 0 else 0.0

        return GameAnalysis(
            home_team=kp_game["Home"],
            away_team=kp_game["Visitor"],
            game_time=game_time,
            kenpom_home_score=kp_game["HomePred"],
            kenpom_away_score=kp_game["VisitorPred"],
            kenpom_spread=kp_spread,
            kenpom_total=kp_total,
            kenpom_home_win_prob=home_win_prob,
            kenpom_tempo=kp_game.get("PredTempo", 0),
            vegas_spread=vegas_spread,
            vegas_total=vegas_total,
            vegas_ml_home=market_odds.ml_home or 0,
            vegas_ml_away=market_odds.ml_away or 0,
            spread_diff=spread_diff,
            total_diff=total_diff,
            value_bets=value_bets,
            home_rank=kp_game.get("HomeRank", 0),
            away_rank=kp_game.get("VisitorRank", 0),
            stat_comparison=stat_comparison,
            major_stat_diffs=stat_comparison.major_differences if stat_comparison else []
        )
    
    def _create_analysis_kenpom_only(self, kp_game: Dict[str, Any]) -> GameAnalysis:
        """Create analysis when odds aren't available."""
        # KenPom spread (negative = home favored, matching betting convention)
        kp_spread = kp_game["VisitorPred"] - kp_game["HomePred"]
        kp_total = kp_game["HomePred"] + kp_game["VisitorPred"]

        # KenPom can return win probability as either decimal (0.95) or percentage (95)
        home_win_prob = self._normalize_win_probability(kp_game.get("HomeWP", 0.5))

        # Parse the game date
        game_date = kp_game.get("DateOfGame", date.today().isoformat())
        game_time = datetime.fromisoformat(game_date) if "T" in game_date else datetime.now()

        # Get team stats and create comparison
        home_stats = self._get_team_stats(kp_game["Home"])
        away_stats = self._get_team_stats(kp_game["Visitor"])
        stat_comparison = self.calculator.compare_team_stats(home_stats, away_stats)

        return GameAnalysis(
            home_team=kp_game["Home"],
            away_team=kp_game["Visitor"],
            game_time=game_time,
            kenpom_home_score=kp_game["HomePred"],
            kenpom_away_score=kp_game["VisitorPred"],
            kenpom_spread=kp_spread,
            kenpom_total=kp_total,
            kenpom_home_win_prob=home_win_prob,
            kenpom_tempo=kp_game.get("PredTempo", 0),
            vegas_spread=0,
            vegas_total=0,
            vegas_ml_home=0,
            vegas_ml_away=0,
            spread_diff=0,
            total_diff=0,
            value_bets=[],
            home_rank=kp_game.get("HomeRank", 0),
            away_rank=kp_game.get("VisitorRank", 0),
            stat_comparison=stat_comparison,
            major_stat_diffs=stat_comparison.major_differences if stat_comparison else []
        )
    
    @staticmethod
    def _is_better_odds(new_odds: int, current_best: Optional[int]) -> bool:
        """
        Determine if new odds are better than current best for American odds.

        For American odds, "better" means higher payout:
        - Negative odds (favorites): -108 is better than -112 (less juice)
        - Positive odds (underdogs): +150 is better than +120 (higher payout)

        To compare properly, we convert to decimal odds where higher is always better.
        """
        if current_best is None:  # No current best
            return True

        # Convert to decimal odds for comparison (higher decimal = better payout)
        def to_decimal(odds: int) -> float:
            if odds > 0:
                return (odds / 100) + 1
            elif odds < 0:
                return (100 / abs(odds)) + 1
            else:
                return 1.0  # Invalid odds, return minimum

        return to_decimal(new_odds) > to_decimal(current_best)

    def _parse_odds(self, odds_game: Dict[str, Any]) -> MarketOdds:
        """Parse odds data from The Odds API response."""
        market = MarketOdds(
            home_team=odds_game["home_team"],
            away_team=odds_game["away_team"]
        )

        # Track best odds (initialized to None to indicate no odds found yet)
        best_spread_home: Optional[int] = None
        best_spread_away: Optional[int] = None
        best_over: Optional[int] = None
        best_under: Optional[int] = None
        best_ml_home: Optional[int] = None
        best_ml_away: Optional[int] = None

        spreads: List[float] = []
        totals: List[float] = []
        ml_home_list: List[int] = []
        ml_away_list: List[int] = []

        for bookmaker in odds_game.get("bookmakers", []):
            book_name = bookmaker["title"]

            for mkt in bookmaker.get("markets", []):
                if mkt["key"] == "spreads":
                    for outcome in mkt["outcomes"]:
                        if outcome["name"] == odds_game["home_team"]:
                            spreads.append(outcome["point"])
                            if self._is_better_odds(outcome["price"], best_spread_home):
                                best_spread_home = outcome["price"]
                                market.best_spread_home_odds = outcome["price"]
                                market.best_spread_home_book = book_name
                                market.spread = outcome["point"]
                        else:
                            if self._is_better_odds(outcome["price"], best_spread_away):
                                best_spread_away = outcome["price"]
                                market.best_spread_away_odds = outcome["price"]
                                market.best_spread_away_book = book_name

                elif mkt["key"] == "totals":
                    for outcome in mkt["outcomes"]:
                        totals.append(outcome.get("point", 0))
                        if outcome["name"] == "Over":
                            if self._is_better_odds(outcome["price"], best_over):
                                best_over = outcome["price"]
                                market.best_over_odds = outcome["price"]
                                market.best_over_book = book_name
                                market.total = outcome.get("point", 0)
                        else:
                            if self._is_better_odds(outcome["price"], best_under):
                                best_under = outcome["price"]
                                market.best_under_odds = outcome["price"]
                                market.best_under_book = book_name

                elif mkt["key"] == "h2h":
                    for outcome in mkt["outcomes"]:
                        if outcome["name"] == odds_game["home_team"]:
                            ml_home_list.append(outcome["price"])
                            if self._is_better_odds(outcome["price"], best_ml_home):
                                best_ml_home = outcome["price"]
                                market.best_ml_home_odds = outcome["price"]
                                market.best_ml_home_book = book_name
                        else:
                            ml_away_list.append(outcome["price"])
                            if self._is_better_odds(outcome["price"], best_ml_away):
                                best_ml_away = outcome["price"]
                                market.best_ml_away_odds = outcome["price"]
                                market.best_ml_away_book = book_name
        
        # Set consensus/average values with safe division
        if spreads:
            market.spread = sum(spreads) / len(spreads)
            market.spread_home_odds = -110  # Standard juice
            market.spread_away_odds = -110

        if totals:
            # Filter out zero values and deduplicate
            valid_totals = [t for t in set(totals) if t > 0]
            if valid_totals:
                market.total = sum(valid_totals) / len(valid_totals)
                market.over_odds = -110
                market.under_odds = -110

        if ml_home_list:
            market.ml_home = int(sum(ml_home_list) / len(ml_home_list))
        if ml_away_list:
            market.ml_away = int(sum(ml_away_list) / len(ml_away_list))

        return market
    
    # ==================== FORMATTING ====================
    
    def format_analysis(self, analysis: GameAnalysis) -> str:
        """Format a game analysis for display."""
        output = []
        
        # Header
        time_str = analysis.game_time.strftime("%I:%M %p")
        output.append(f"\n{'='*60}")
        output.append(f"🏀 #{analysis.away_rank} {analysis.away_team} @ #{analysis.home_rank} {analysis.home_team}")
        output.append(f"⏰ {time_str}")
        output.append(f"{'='*60}")
        
        # KenPom Predictions
        output.append(f"\n📊 KenPom Prediction:")
        output.append(f"   Score: {analysis.away_team} {analysis.kenpom_away_score:.1f} - {analysis.home_team} {analysis.kenpom_home_score:.1f}")
        output.append(f"   Spread: {analysis.home_team} {-analysis.kenpom_spread:+.1f}")
        output.append(f"   Total: {analysis.kenpom_total:.1f}")
        output.append(f"   Win Prob: {analysis.home_team} {analysis.kenpom_home_win_prob:.1%}")
        
        # Vegas Lines
        if analysis.vegas_spread != 0:
            output.append(f"\n💰 Vegas Lines:")
            output.append(f"   Spread: {analysis.home_team} {analysis.vegas_spread:+.1f}")
            output.append(f"   Total: {analysis.vegas_total:.1f}")
            output.append(f"   Moneyline: {analysis.home_team} {analysis.vegas_ml_home:+d} / {analysis.away_team} {analysis.vegas_ml_away:+d}")
        
        # Line Comparison
        if analysis.spread_diff != 0:
            output.append(f"\n📈 Line Comparison:")
            output.append(f"   Spread Diff: {analysis.spread_diff:+.1f} (KenPom - Vegas)")
            output.append(f"   Total Diff: {analysis.total_diff:+.1f}")
        
        # Value Bets
        if analysis.value_bets:
            output.append(f"\n🎯 VALUE BETS FOUND:")
            for vb in analysis.value_bets:
                output.append(f"\n{self.calculator.format_value_bet(vb)}")
        else:
            output.append(f"\n❌ No value bets identified")
        
        return "\n".join(output)
