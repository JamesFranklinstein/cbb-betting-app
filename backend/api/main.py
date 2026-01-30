"""
FastAPI Backend

REST API for the CBB Betting application.
"""

import asyncio
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from clients import KenPomClient, OddsAPIClient
from services import GameService
from services.bet_history import BetHistoryService
from ml import MLPredictor
from models import init_db, get_db, SessionLocal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# CORS configuration - set allowed origins from environment or use defaults
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000,https://frontend-indol-omega-55.vercel.app,https://cbb-betting-app.vercel.app,https://indianriver.ai,https://www.indianriver.ai"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")

# Check if running on Vercel (serverless)
IS_SERVERLESS = os.getenv("VERCEL", "false").lower() == "1" or os.getenv("AWS_LAMBDA_FUNCTION_NAME") is not None


# ==================== PYDANTIC MODELS ====================

class ValueBetResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    home_team: str
    away_team: str
    bet_type: str
    side: str
    model_prob: float
    market_odds: int
    market_implied_prob: float
    edge: float
    ev: float
    kelly_fraction: float
    recommended_book: str
    confidence: str
    market_line: Optional[float] = None
    model_line: Optional[float] = None


class TeamStatsResponse(BaseModel):
    """Full team statistics from KenPom."""
    team_name: str
    rank: int = 0
    adj_em: float = 0.0
    adj_oe: float = 0.0
    adj_de: float = 0.0
    adj_tempo: float = 0.0
    efg_pct: float = 0.0
    to_pct: float = 0.0
    or_pct: float = 0.0
    ft_rate: float = 0.0
    d_efg_pct: float = 0.0
    d_to_pct: float = 0.0
    d_or_pct: float = 0.0
    d_ft_rate: float = 0.0
    sos: float = 0.0
    luck: float = 0.0


class StatDifferenceResponse(BaseModel):
    """Statistical difference between two teams."""
    stat_name: str
    display_name: str
    home_value: float
    away_value: float
    difference: float
    advantage: str  # "home", "away", or "neutral"
    significance: str  # "major", "moderate", "minor"
    higher_is_better: bool = True


class TeamStatComparisonResponse(BaseModel):
    """Full statistical comparison between two teams."""
    home_stats: TeamStatsResponse
    away_stats: TeamStatsResponse
    stat_differences: List[StatDifferenceResponse]
    major_differences: List[StatDifferenceResponse]
    efficiency_edge: str
    tempo_mismatch: bool
    shooting_edge: str
    rebounding_edge: str
    turnover_edge: str


class DataQualityWarningResponse(BaseModel):
    """Warning about potential data quality issues."""
    code: str
    severity: str
    message: str
    details: dict = {}


class GameAnalysisResponse(BaseModel):
    home_team: str
    away_team: str
    game_time: datetime
    home_rank: int
    away_rank: int

    kenpom_home_score: float
    kenpom_away_score: float
    kenpom_spread: float
    kenpom_total: float
    kenpom_home_win_prob: float
    kenpom_tempo: float

    vegas_spread: float
    vegas_total: float
    vegas_ml_home: int
    vegas_ml_away: int

    spread_diff: float
    total_diff: float

    value_bets: List[ValueBetResponse]

    # Team statistics comparison
    stat_comparison: Optional[TeamStatComparisonResponse] = None
    major_stat_diffs: List[StatDifferenceResponse] = []

    # Data quality warnings
    data_warnings: List[DataQualityWarningResponse] = []


class TeamRatingResponse(BaseModel):
    team_name: str
    rank: int
    conference: str
    wins: int
    losses: int
    adj_em: float
    adj_oe: float
    adj_de: float
    adj_tempo: float
    sos: float


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


# ==================== APP SETUP ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and services on startup."""
    init_db()
    logger.info("[Startup] Database initialized")
    yield


app = FastAPI(
    title="CBB Betting API",
    description="College Basketball betting analysis and value bet detection",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - restricted to configured origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Service instances
game_service = GameService()
ml_predictor = MLPredictor()
bet_history_service = BetHistoryService()


# ==================== ENDPOINTS ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version="1.0.0"
    )


@app.get("/api/games/today", response_model=List[GameAnalysisResponse])
async def get_todays_games():
    """Get analysis for all of today's games."""
    try:
        analyses = await game_service.get_todays_analysis()
        return [_analysis_to_response(a) for a in analyses]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/games/{game_date}", response_model=List[GameAnalysisResponse])
async def get_games_by_date(game_date: str):
    """Get analysis for games on a specific date (YYYY-MM-DD)."""
    try:
        # Validate date format
        datetime.strptime(game_date, "%Y-%m-%d")

        # Use the service to get analyses for the specific date
        analyses = await game_service.get_analysis_for_date(game_date)
        return [_analysis_to_response(a) for a in analyses]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/value-bets", response_model=List[dict])
async def get_value_bets(
    min_edge: float = Query(0.03, ge=0.0, le=1.0, description="Minimum edge to include (0.0 to 1.0)"),
    bet_type: Optional[str] = Query(None, description="Filter by bet type: spread, total, moneyline")
):
    """Get all value bets for today's games, sorted by confidence (high first)."""
    try:
        value_games = await game_service.get_value_bets_today(min_edge)

        results = []
        for analysis, value_bets in value_games:
            # Build major stat diffs list for this game
            major_stat_diffs = []
            if analysis.major_stat_diffs:
                for diff in analysis.major_stat_diffs:
                    major_stat_diffs.append({
                        "stat_name": diff.stat_name,
                        "display_name": diff.display_name,
                        "home_value": diff.home_value,
                        "away_value": diff.away_value,
                        "difference": diff.difference,
                        "advantage": diff.advantage,
                        "significance": diff.significance
                    })

            # Build stat comparison summary
            stat_summary = None
            if analysis.stat_comparison:
                stat_summary = {
                    "efficiency_edge": analysis.stat_comparison.efficiency_edge,
                    "shooting_edge": analysis.stat_comparison.shooting_edge,
                    "rebounding_edge": analysis.stat_comparison.rebounding_edge,
                    "turnover_edge": analysis.stat_comparison.turnover_edge,
                    "tempo_mismatch": analysis.stat_comparison.tempo_mismatch
                }

            # Build data warnings list
            data_warnings = []
            if analysis.data_warnings:
                for w in analysis.data_warnings:
                    data_warnings.append({
                        "code": w.code,
                        "severity": w.severity,
                        "message": w.message,
                        "details": w.details
                    })

            for vb in value_bets:
                if bet_type is None or vb.bet_type.value == bet_type:
                    results.append({
                        "home_team": analysis.home_team,
                        "away_team": analysis.away_team,
                        "home_rank": analysis.home_rank,
                        "away_rank": analysis.away_rank,
                        "game_time": analysis.game_time.isoformat(),
                        "kenpom_spread": analysis.kenpom_spread,
                        "kenpom_total": analysis.kenpom_total,
                        "vegas_spread": analysis.vegas_spread,
                        "vegas_total": analysis.vegas_total,
                        "spread_diff": analysis.spread_diff,
                        "total_diff": analysis.total_diff,
                        "bet": {
                            "type": vb.bet_type.value,
                            "side": vb.side,
                            "odds": vb.market_odds,
                            "line": vb.market_line,
                            "book": vb.recommended_book,
                            "edge": vb.edge,
                            "ev": vb.ev,
                            "kelly": vb.kelly_fraction,
                            "confidence": vb.confidence,
                            "confidence_score": vb.confidence_score,
                            "confidence_factors": vb.confidence_factors
                        },
                        "major_stat_diffs": major_stat_diffs,
                        "stat_summary": stat_summary,
                        "data_warnings": data_warnings
                    })

        # Sort by confidence: high first, then medium, then low
        confidence_order = {"high": 0, "medium": 1, "low": 2}
        results.sort(key=lambda x: (
            confidence_order.get(x["bet"]["confidence"], 3),  # Primary: confidence tier
            -x["bet"]["confidence_score"]  # Secondary: score within tier (higher first)
        ))

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/matchup")
async def analyze_matchup(
    home: str = Query(..., description="Home team name"),
    away: str = Query(..., description="Away team name"),
    game_date: Optional[str] = Query(None, description="Game date (YYYY-MM-DD)")
):
    """Analyze a specific matchup."""
    try:
        analysis = await game_service.get_game_analysis(home, away, game_date)
        if not analysis:
            raise HTTPException(status_code=404, detail="Game not found")
        return _analysis_to_response(analysis)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/rankings", response_model=List[TeamRatingResponse])
async def get_rankings(
    year: int = Query(2025, description="Season year"),
    conference: Optional[str] = Query(None, description="Conference short name"),
    limit: int = Query(50, description="Number of teams to return")
):
    """Get KenPom rankings."""
    try:
        kenpom = KenPomClient()
        ratings = await kenpom.get_ratings(year=year, conference=conference)
        
        # Sort by rank
        ratings.sort(key=lambda x: x.get("RankAdjEM", 999))
        ratings = ratings[:limit]
        
        return [
            TeamRatingResponse(
                team_name=r.get("TeamName", ""),
                rank=r.get("RankAdjEM", 0),
                conference=r.get("ConfShort", ""),
                wins=r.get("Wins", 0),
                losses=r.get("Losses", 0),
                adj_em=r.get("AdjEM", 0),
                adj_oe=r.get("AdjOE", 0),
                adj_de=r.get("AdjDE", 0),
                adj_tempo=r.get("AdjTempo", 0),
                sos=r.get("SOS", 0)
            )
            for r in ratings
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/spreads")
async def get_spread_discrepancies(
    min_diff: float = Query(2.0, description="Minimum spread difference")
):
    """Get games where KenPom spread differs significantly from Vegas."""
    try:
        analyses = await game_service.get_todays_analysis()
        
        spread_games = [
            {
                "home_team": a.home_team,
                "away_team": a.away_team,
                "game_time": a.game_time.isoformat(),
                "kenpom_spread": a.kenpom_spread,
                "vegas_spread": a.vegas_spread,
                "difference": a.spread_diff,
                "lean": f"{a.away_team}" if a.spread_diff > 0 else f"{a.home_team}"
            }
            for a in analyses
            if a.vegas_spread != 0 and abs(a.spread_diff) >= min_diff
        ]
        
        spread_games.sort(key=lambda x: abs(x["difference"]), reverse=True)
        return spread_games
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/totals")
async def get_total_discrepancies(
    min_diff: float = Query(3.0, description="Minimum total difference")
):
    """Get games where KenPom total differs significantly from Vegas."""
    try:
        analyses = await game_service.get_todays_analysis()
        
        total_games = [
            {
                "home_team": a.home_team,
                "away_team": a.away_team,
                "game_time": a.game_time.isoformat(),
                "kenpom_total": a.kenpom_total,
                "vegas_total": a.vegas_total,
                "difference": a.total_diff,
                "lean": "over" if a.total_diff > 0 else "under"
            }
            for a in analyses
            if a.vegas_total != 0 and abs(a.total_diff) >= min_diff
        ]
        
        total_games.sort(key=lambda x: abs(x["difference"]), reverse=True)
        return total_games
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/odds/usage")
async def get_odds_api_usage():
    """Get The Odds API usage statistics."""
    try:
        odds = OddsAPIClient()
        # Make a lightweight request to get usage headers
        await odds.get_sports()
        return odds.get_api_usage()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HELPER FUNCTIONS ====================

async def _auto_update_pending_results():
    """
    Automatically fetch scores and update pending bet results.
    Called when fetching bet history to keep results current.
    """
    # Get pending bets
    pending = bet_history_service.get_pending_bets()

    if not pending:
        return 0

    # Check if any pending bets are from games that should be completed
    # (games from yesterday or earlier)
    # Use UTC for consistent timezone handling (server may run in different TZ)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    has_old_pending = any(bet.date < today for bet in pending)
    if not has_old_pending:
        return 0

    # Try to get scores from Odds API (completed events)
    odds_client = OddsAPIClient()

    try:
        scores_data = await odds_client.get_scores(days_from=3)
    except Exception as e:
        logger.warning(f"Could not fetch scores: {e}")
        return 0

    # Build scores dict with multiple key formats for matching
    # Include game date to prevent matching wrong games
    scores = {}
    for game in scores_data:
        if game.get("completed"):
            home_team = game.get("home_team")
            away_team = game.get("away_team")

            # Extract game date from commence_time
            commence_time = game.get("commence_time", "")
            game_date = None
            if commence_time:
                try:
                    # Parse ISO format and extract date
                    game_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                    game_date = game_dt.strftime("%Y-%m-%d")
                except (ValueError, AttributeError):
                    pass

            home_score = None
            away_score = None

            for score in game.get("scores", []):
                if score.get("name") == home_team:
                    home_score = int(score.get("score", 0))
                elif score.get("name") == away_team:
                    away_score = int(score.get("score", 0))

            # Validate scores - must be positive for a completed basketball game
            if home_score is not None and away_score is not None and home_score > 0 and away_score > 0:
                score_data = {"home": home_score, "away": away_score, "game_date": game_date}
                # Store with exact team names
                key = f"{home_team} vs {away_team}"
                scores[key] = score_data
                # Also store normalized versions for fuzzy matching
                scores[_normalize_team_key(home_team, away_team)] = score_data

    # Match pending bets to scores using fuzzy matching
    updated = 0
    for bet in pending:
        if bet.result is not None:
            continue

        # Try exact match first
        game_key = f"{bet.home_team} vs {bet.away_team}"
        if game_key in scores:
            score = scores[game_key]
        else:
            # Try normalized match
            normalized_key = _normalize_team_key(bet.home_team, bet.away_team)
            if normalized_key in scores:
                score = scores[normalized_key]
            else:
                # Try to find a fuzzy match
                score = _find_matching_score(bet.home_team, bet.away_team, scores)
                if not score:
                    continue

        # CRITICAL: Validate game date matches bet date to prevent grading wrong games
        # This prevents matching yesterday's "Penn St vs Ohio St" to today's different game
        score_game_date = score.get("game_date")
        if score_game_date and bet.date:
            if score_game_date != bet.date:
                logger.warning(
                    f"Skipping score match for {bet.home_team} vs {bet.away_team}: "
                    f"score date {score_game_date} != bet date {bet.date}"
                )
                continue

        # Update the bet with the score
        updated += bet_history_service.update_single_bet_result(
            bet.bet_id,
            score["home"],
            score["away"]
        )

    if updated > 0:
        logger.info(f"Auto-updated {updated} bet results")

    return updated


def _normalize_team_key(home: str, away: str) -> str:
    """Normalize team names for matching."""
    def normalize(name: str) -> str:
        # Remove common suffixes and normalize
        name = name.lower().strip()
        # Remove state abbreviations like "St." or "State"
        name = name.replace(" st.", " state").replace(" st ", " state ")
        # Remove "university" and common words
        for word in ["university", "college", "the "]:
            name = name.replace(word, "")
        return name.strip()

    return f"{normalize(home)} vs {normalize(away)}"


# Common team name mappings (KenPom name -> Odds API variations)
TEAM_NAME_ALIASES = {
    "miami fl": ["miami", "miami hurricanes"],
    "miami oh": ["miami (oh)", "miami redhawks", "miami ohio"],
    "florida st.": ["florida state", "florida st", "fsu"],
    "michigan st.": ["michigan state", "michigan st", "msu"],
    "ohio st.": ["ohio state", "ohio st", "osu"],
    "penn st.": ["penn state", "penn st", "psu"],
    "st. john's": ["st john's", "st johns", "saint john's"],
    "st. bonaventure": ["st bonaventure", "saint bonaventure"],
    "texas a&m": ["texas am", "texas a&m aggies"],
    "unc": ["north carolina", "tar heels"],
    "uconn": ["connecticut", "huskies"],
    "lsu": ["louisiana state"],
    "ucf": ["central florida", "knights"],
    "unlv": ["nevada las vegas"],
    "utep": ["texas el paso"],
    "vcu": ["virginia commonwealth"],
    "smu": ["southern methodist"],
    "tcu": ["texas christian"],
    "byu": ["brigham young"],
}


def _get_team_variations(team_name: str) -> List[str]:
    """Get all possible variations of a team name for matching."""
    team_lower = team_name.lower().strip()
    variations = [team_lower]

    # Add aliases if they exist
    if team_lower in TEAM_NAME_ALIASES:
        variations.extend(TEAM_NAME_ALIASES[team_lower])

    # Also check if team_lower is contained in any alias key
    for key, aliases in TEAM_NAME_ALIASES.items():
        if key in team_lower or team_lower in key:
            variations.extend(aliases)
            variations.append(key)

    # Normalize St./State variations
    if " st." in team_lower:
        variations.append(team_lower.replace(" st.", " state"))
    if " state" in team_lower:
        variations.append(team_lower.replace(" state", " st."))
        variations.append(team_lower.replace(" state", " st"))

    return list(set(variations))


def _find_matching_score(home_team: str, away_team: str, scores: dict) -> Optional[dict]:
    """Find a matching score using improved fuzzy matching on team names.

    IMPORTANT: Must match BOTH teams distinctly - can't match same API team to both bet teams.
    """
    home_variations = _get_team_variations(home_team)
    away_variations = _get_team_variations(away_team)

    for key, score in scores.items():
        key_lower = key.lower()

        # Split the key into home and away parts (format: "Team1 vs Team2")
        if " vs " not in key_lower:
            continue
        api_home_part, api_away_part = key_lower.split(" vs ", 1)

        # Check if home team matches EITHER part of the API game
        home_matches_api_home = False
        home_matches_api_away = False
        for var in home_variations:
            var_words = var.split()
            primary_words = [w for w in var_words if len(w) > 3]
            if primary_words:
                if any(w in api_home_part for w in primary_words):
                    home_matches_api_home = True
                if any(w in api_away_part for w in primary_words):
                    home_matches_api_away = True

        # Check if away team matches EITHER part of the API game
        away_matches_api_home = False
        away_matches_api_away = False
        for var in away_variations:
            var_words = var.split()
            primary_words = [w for w in var_words if len(w) > 3]
            if primary_words:
                if any(w in api_home_part for w in primary_words):
                    away_matches_api_home = True
                if any(w in api_away_part for w in primary_words):
                    away_matches_api_away = True

        # Valid match: home matches one side AND away matches the OTHER side
        # (home->api_home AND away->api_away) OR (home->api_away AND away->api_home)
        valid_match = (
            (home_matches_api_home and away_matches_api_away) or
            (home_matches_api_away and away_matches_api_home)
        )

        if valid_match:
            return score

    return None


# ==================== BET HISTORY ENDPOINTS ====================

@app.get("/api/bet-history")
async def get_bet_history(
    days: int = Query(30, description="Number of days of history to return"),
    auto_update: bool = Query(True, description="Auto-fetch and update results for pending bets")
):
    """Get bet history with results for the last N days."""
    try:
        # Auto-update pending results if requested
        if auto_update:
            try:
                await _auto_update_pending_results()
            except Exception as e:
                logger.warning(f"Auto-update of results failed: {e}")

        history = bet_history_service.get_history_for_display(days)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bet-history/stats")
async def get_betting_stats():
    """Get overall betting statistics and performance."""
    try:
        stats = bet_history_service.get_overall_stats()
        return {
            "total_bets": stats.total_bets,
            "total_wins": stats.total_wins,
            "total_losses": stats.total_losses,
            "total_pushes": stats.total_pushes,
            "total_pending": stats.total_pending,
            "win_rate": stats.win_rate,
            "high_conf": {
                "record": stats.high_conf_record,
                "win_rate": stats.high_conf_win_rate
            },
            "medium_conf": {
                "record": stats.medium_conf_record,
                "win_rate": stats.medium_conf_win_rate
            },
            "low_conf": {
                "record": stats.low_conf_record,
                "win_rate": stats.low_conf_win_rate
            },
            "by_type": {
                "spread": {
                    "record": stats.spread_record,
                    "win_rate": stats.spread_win_rate
                },
                "total": {
                    "record": stats.total_record,
                    "win_rate": stats.total_win_rate
                },
                "moneyline": {
                    "record": stats.ml_record,
                    "win_rate": stats.ml_win_rate
                }
            },
            "avg_edge": stats.avg_edge,
            "avg_ev": stats.avg_ev,
            "streaks": {
                "current": stats.current_streak,
                "type": stats.streak_type,
                "longest_win": stats.longest_win_streak,
                "longest_loss": stats.longest_loss_streak
            },
            "recent": {
                "last_7_days": {
                    "record": stats.last_7_days_record,
                    "win_rate": stats.last_7_days_win_rate
                },
                "last_30_days": {
                    "record": stats.last_30_days_record,
                    "win_rate": stats.last_30_days_win_rate
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bet-history/daily/{date}")
async def get_daily_summary(date: str):
    """Get betting summary for a specific date (YYYY-MM-DD)."""
    try:
        # Validate date format
        datetime.strptime(date, "%Y-%m-%d")
        summary = bet_history_service.get_daily_summary(date)
        return {
            "date": summary.date,
            "total_bets": summary.total_bets,
            "wins": summary.wins,
            "losses": summary.losses,
            "pushes": summary.pushes,
            "pending": summary.pending,
            "win_rate": summary.win_rate,
            "by_confidence": {
                "high": {
                    "bets": summary.high_conf_bets,
                    "wins": summary.high_conf_wins
                },
                "medium": {
                    "bets": summary.medium_conf_bets,
                    "wins": summary.medium_conf_wins
                },
                "low": {
                    "bets": summary.low_conf_bets,
                    "wins": summary.low_conf_wins
                }
            },
            "by_type": {
                "spread": {
                    "bets": summary.spread_bets,
                    "wins": summary.spread_wins
                },
                "total": {
                    "bets": summary.total_bets_count,
                    "wins": summary.total_wins
                },
                "moneyline": {
                    "bets": summary.ml_bets,
                    "wins": summary.ml_wins
                }
            },
            "avg_edge": summary.avg_edge,
            "avg_ev": summary.avg_ev
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bet-history/store")
async def store_todays_bets(
    min_edge: float = Query(0.03, description="Minimum edge to store")
):
    """Store today's medium and high confidence value bets for tracking."""
    try:
        # Get today's value bets
        value_games = await game_service.get_value_bets_today(min_edge)

        # Format for storage - only medium and high confidence bets
        value_bets_data = []
        skipped_low = 0
        for analysis, value_bets in value_games:
            for vb in value_bets:
                # Only track medium and high confidence bets
                if vb.confidence in ("medium", "high"):
                    value_bets_data.append({
                        "home_team": analysis.home_team,
                        "away_team": analysis.away_team,
                        "game_time": analysis.game_time.isoformat(),
                        "home_rank": analysis.home_rank,
                        "away_rank": analysis.away_rank,
                        "kenpom_spread": analysis.kenpom_spread,
                        "kenpom_total": analysis.kenpom_total,
                        "vegas_spread": analysis.vegas_spread,
                        "vegas_total": analysis.vegas_total,
                        "bet": {
                            "type": vb.bet_type.value,
                            "side": vb.side,
                            "line": vb.market_line,
                            "odds": vb.market_odds,
                            "book": vb.recommended_book,
                            "model_prob": vb.model_prob,
                            "market_implied_prob": vb.market_implied_prob,
                            "edge": vb.edge,
                            "ev": vb.ev,
                            "kelly": vb.kelly_fraction,
                            "confidence": vb.confidence,
                            "confidence_score": vb.confidence_score
                        }
                    })
                else:
                    skipped_low += 1

        # Store bets
        count = bet_history_service.store_all_value_bets(value_bets_data)

        return {
            "stored": count,
            "total_trackable": len(value_bets_data),
            "skipped_low_confidence": skipped_low,
            "message": f"Stored {count} new medium/high confidence bets (skipped {skipped_low} low confidence)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bet-history/update-results")
async def update_bet_results(scores: dict):
    """
    Update bet results with final scores.

    Body should be: {"Team A vs Team B": {"home": 75, "away": 68}, ...}
    """
    try:
        updated = bet_history_service.update_results(scores)
        return {
            "updated": updated,
            "message": f"Updated {updated} bet results"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bet-history/pending")
async def get_pending_bets():
    """Get all bets awaiting results."""
    try:
        pending = bet_history_service.get_pending_bets()
        return [
            {
                "id": bet.id,
                "date": bet.date,
                "game_time": bet.game_time,
                "home_team": bet.home_team,
                "away_team": bet.away_team,
                "bet_type": bet.bet_type,
                "side": bet.side,
                "line": bet.line,
                "odds": bet.odds,
                "confidence": bet.confidence
            }
            for bet in pending
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bet-history/debug/all-dates")
async def debug_all_bet_dates():
    """Debug endpoint to see all unique dates with bet counts."""
    try:
        all_bets = bet_history_service.get_all_bets()
        date_counts = {}
        for bet in all_bets:
            if bet.date not in date_counts:
                date_counts[bet.date] = {"total": 0, "pending": 0, "graded": 0}
            date_counts[bet.date]["total"] += 1
            if bet.result is None:
                date_counts[bet.date]["pending"] += 1
            else:
                date_counts[bet.date]["graded"] += 1

        return {
            "total_bets": len(all_bets),
            "dates": dict(sorted(date_counts.items(), reverse=True))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bet-history/debug/match-test")
async def debug_match_test():
    """Debug endpoint to test team name matching between pending bets and Odds API scores."""
    try:
        pending = bet_history_service.get_pending_bets()
        if not pending:
            return {"message": "No pending bets"}

        odds_client = OddsAPIClient()
        scores_data = await odds_client.get_scores(days_from=3)

        # Build scores dict
        scores = {}
        raw_games = []
        for game in scores_data:
            if game.get("completed"):
                home_team = game.get("home_team")
                away_team = game.get("away_team")
                raw_games.append({"home": home_team, "away": away_team})

                home_score = None
                away_score = None
                for score in game.get("scores", []):
                    if score.get("name") == home_team:
                        home_score = int(score.get("score", 0))
                    elif score.get("name") == away_team:
                        away_score = int(score.get("score", 0))

                if home_score is not None and away_score is not None:
                    score_data = {"home": home_score, "away": away_score}
                    key = f"{home_team} vs {away_team}"
                    scores[key] = score_data
                    scores[_normalize_team_key(home_team, away_team)] = score_data

        # Check each pending bet for matches
        match_results = []
        for bet in pending[:10]:  # Limit to first 10 for readability
            bet_key = f"{bet.home_team} vs {bet.away_team}"
            normalized_bet_key = _normalize_team_key(bet.home_team, bet.away_team)

            # Check exact match
            exact_match = bet_key in scores
            normalized_match = normalized_bet_key in scores

            # Check fuzzy match
            fuzzy_result = _find_matching_score(bet.home_team, bet.away_team, scores)

            match_results.append({
                "bet_teams": bet_key,
                "normalized_bet": normalized_bet_key,
                "exact_match": exact_match,
                "normalized_match": normalized_match,
                "fuzzy_match_found": fuzzy_result is not None,
                "fuzzy_score": fuzzy_result
            })

        return {
            "pending_bets": len(pending),
            "completed_games_from_api": len(raw_games),
            "sample_api_games": raw_games[:10],
            "score_keys": list(scores.keys())[:20],
            "match_results": match_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bet-history/reset/{bet_id}")
async def reset_bet_result(bet_id: str):
    """Reset a bet's result back to pending (for fixing incorrect grades)."""
    try:
        session = SessionLocal()
        try:
            from models.database import StoredBet
            bet = session.query(StoredBet).filter(StoredBet.bet_id == bet_id).first()
            if not bet:
                raise HTTPException(status_code=404, detail=f"Bet not found: {bet_id}")

            old_result = bet.result
            bet.result = None
            bet.home_score = None
            bet.away_score = None
            bet.actual_margin = None
            bet.actual_total = None
            session.commit()

            return {
                "status": "success",
                "bet_id": bet_id,
                "old_result": old_result,
                "message": f"Bet reset to pending"
            }
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/bet-history/debug/{date}")
async def debug_bets_by_date(date: str):
    """Debug endpoint to check bets for a specific date (format: YYYY-MM-DD)."""
    try:
        bets = bet_history_service.get_bets_by_date(date)
        return {
            "date": date,
            "total_bets": len(bets),
            "bets": [
                {
                    "id": bet.bet_id,
                    "date": bet.date,
                    "game_time": bet.game_time,
                    "home_team": bet.home_team,
                    "away_team": bet.away_team,
                    "bet_type": bet.bet_type,
                    "side": bet.side,
                    "line": bet.line,
                    "odds": bet.odds,
                    "confidence": bet.confidence,
                    "result": bet.result,
                    "home_score": bet.home_score,
                    "away_score": bet.away_score,
                    "actual_margin": bet.actual_margin,
                    "settled_at": str(bet.settled_at) if bet.settled_at else None,
                    "created_at": str(bet.created_at) if bet.created_at else None
                }
                for bet in bets
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== ML MANAGEMENT ENDPOINTS ====================

class MLVersionResponse(BaseModel):
    """ML Model version information."""
    model_config = {"protected_namespaces": ()}

    version: str
    model_type: str
    brier_score: Optional[float] = None
    accuracy: Optional[float] = None
    spread_mae: Optional[float] = None
    total_mae: Optional[float] = None
    live_accuracy: Optional[float] = None
    live_roi: Optional[float] = None
    total_predictions: Optional[int] = None
    is_active: bool
    trained_at: Optional[str] = None
    model_path: Optional[str] = None


class MLCalibrationResponse(BaseModel):
    """ML Model calibration metrics."""
    model_config = {"protected_namespaces": ()}

    model_version: Optional[str] = None
    temperature: Optional[float] = None
    calibration_buckets: List[dict] = []
    overall_calibration_error: Optional[float] = None
    brier_score: Optional[float] = None


class MLFeedbackStatsResponse(BaseModel):
    """ML Feedback loop statistics."""
    total_training_samples: int = 0
    samples_last_7_days: int = 0
    samples_last_30_days: int = 0
    accuracy_stats: dict = {}
    avg_errors: dict = {}


@app.get("/api/ml/versions", response_model=List[MLVersionResponse])
async def get_ml_versions(
    limit: int = Query(10, description="Maximum versions to return")
):
    """Get list of ML model versions with performance metrics."""
    try:
        from ml.version_manager import ModelVersionManager

        session = SessionLocal()
        try:
            version_manager = ModelVersionManager(model_dir="./models", session=session)
            versions = version_manager.list_versions(limit=limit)

            return [
                MLVersionResponse(
                    version=v['version'],
                    model_type=v['model_type'],
                    brier_score=v.get('brier_score'),
                    accuracy=v.get('accuracy'),
                    spread_mae=v.get('spread_mae'),
                    total_mae=v.get('total_mae'),
                    live_accuracy=v.get('live_accuracy'),
                    live_roi=v.get('live_roi'),
                    total_predictions=v.get('total_predictions'),
                    is_active=v['is_active'],
                    trained_at=v.get('trained_at'),
                    model_path=v.get('model_path')
                )
                for v in versions
            ]
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error fetching ML versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/versions/active", response_model=Optional[MLVersionResponse])
async def get_active_ml_version():
    """Get the currently active ML model version."""
    try:
        from ml.version_manager import ModelVersionManager

        session = SessionLocal()
        try:
            version_manager = ModelVersionManager(model_dir="./models", session=session)
            active = version_manager.get_active_version()

            if not active:
                return None

            return MLVersionResponse(
                version=active.version,
                model_type=active.model_type,
                brier_score=active.brier_score,
                accuracy=active.val_accuracy,
                spread_mae=active.spread_mae,
                total_mae=active.total_mae,
                live_accuracy=active.live_accuracy,
                live_roi=active.live_roi,
                total_predictions=active.total_predictions,
                is_active=active.is_active,
                trained_at=active.trained_at.isoformat() if active.trained_at else None,
                model_path=active.model_path
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error fetching active ML version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/versions/{version}/activate")
async def activate_ml_version(version: str):
    """Activate a specific ML model version."""
    try:
        from ml.version_manager import ModelVersionManager

        session = SessionLocal()
        try:
            version_manager = ModelVersionManager(model_dir="./models", session=session)
            success = version_manager.activate_version(version)

            if not success:
                raise HTTPException(status_code=404, detail=f"Version {version} not found")

            return {"message": f"Activated model version {version}", "version": version}
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating ML version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/versions/compare")
async def compare_ml_versions(
    version_a: str = Query(..., description="First version to compare"),
    version_b: str = Query(..., description="Second version to compare")
):
    """Compare two ML model versions."""
    try:
        from ml.version_manager import ModelVersionManager

        session = SessionLocal()
        try:
            version_manager = ModelVersionManager(model_dir="./models", session=session)
            comparison = version_manager.compare_versions(version_a, version_b)

            if 'error' in comparison:
                raise HTTPException(status_code=404, detail=comparison['error'])

            return comparison
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing ML versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/data/analyze")
async def analyze_training_data():
    """Analyze training data for quality issues (false results, invalid scores, etc.)."""
    try:
        from ml.data_cleaner import DataCleaner

        with DataCleaner() as cleaner:
            bet_issues = cleaner.analyze_stored_bets()
            training_issues = cleaner.analyze_training_data()

            return {
                "stored_bets": {
                    "total": bet_issues["total_bets"],
                    "graded": bet_issues["graded_bets"],
                    "pending": bet_issues["pending_bets"],
                    "issues": {
                        "future_graded": len(bet_issues["future_graded"]),
                        "invalid_scores": len(bet_issues["invalid_scores"]),
                        "inconsistent_games": len(bet_issues["inconsistent_games"]),
                    },
                    "future_graded_sample": bet_issues["future_graded"][:5],
                    "invalid_scores_sample": bet_issues["invalid_scores"][:5],
                    "inconsistent_games_sample": bet_issues["inconsistent_games"][:5],
                },
                "training_data": {
                    "total": training_issues["total_records"],
                    "issues": {
                        "missing_scores": len(training_issues["missing_scores"]),
                        "invalid_scores": len(training_issues["invalid_scores"]),
                        "missing_features": len(training_issues["missing_features"]),
                        "large_errors": len(training_issues["large_errors"]),
                    }
                }
            }
    except Exception as e:
        logger.error(f"Error analyzing training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/data/clean")
async def clean_training_data(
    dry_run: bool = Query(True, description="If true, only report what would be cleaned")
):
    """Clean training data by removing false results and invalid records."""
    try:
        from ml.data_cleaner import DataCleaner

        with DataCleaner() as cleaner:
            results = cleaner.clean_all(dry_run=dry_run)

            return {
                "dry_run": dry_run,
                "actions": {
                    "future_graded_reset": len(results["future_graded_reset"]),
                    "invalid_scores_reset": len(results["invalid_scores_reset"]),
                    "inconsistent_games_reset": len(results["inconsistent_games_reset"]),
                    "invalid_training_deleted": results["invalid_training_deleted"],
                },
                "summary": results["summary"],
                "message": "Dry run - no changes made" if dry_run else "Data cleaned successfully"
            }
    except Exception as e:
        logger.error(f"Error cleaning training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/retrain/xgboost")
async def trigger_xgboost_retraining(
    min_samples: int = Query(100, description="Minimum training samples required")
):
    """Train a new XGBoost model (recommended over PyTorch for tabular data)."""
    try:
        from ml.feedback_collector import FeedbackCollector
        from ml.xgboost_model import XGBoostPredictor, FEATURE_COLUMNS
        from ml.version_manager import ModelVersionManager

        session = SessionLocal()
        try:
            # Get training data
            collector = FeedbackCollector(session)
            df = collector.get_training_dataframe()

            if len(df) < min_samples:
                return {
                    "status": "skipped",
                    "reason": "insufficient_data",
                    "available_samples": len(df),
                    "required_samples": min_samples
                }

            # Split data (time-based)
            df = df.sort_values('game_date')
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]

            logger.info(f"Training XGBoost on {len(train_df)} games, validating on {len(val_df)} games")

            # Train new model
            predictor = XGBoostPredictor(model_dir="./ml_models")
            metrics = predictor.train(train_df, val_df)

            # Save model
            version_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            model_path = predictor.save(version_str)

            # Get feature importance
            importance = predictor.get_feature_importance()

            # Register version in database
            version_manager = ModelVersionManager("./ml_models", session)
            version = version_manager.create_version(
                model_type='XGBoost',
                features=FEATURE_COLUMNS,
                hyperparameters={
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.05,
                },
                training_metrics={
                    'brier_score': metrics['win']['brier_score'],
                    'accuracy': metrics['win']['accuracy'],
                    'spread_mae': metrics['spread']['mae'],
                    'total_mae': metrics['total']['mae'],
                },
                model_path=model_path
            )

            return {
                "status": "success",
                "version": f"xgboost_v{version_str}",
                "model_path": model_path,
                "metrics": metrics,
                "training_samples": len(train_df),
                "validation_samples": len(val_df),
                "top_features": {
                    "win": list(importance['win'].items())[:5],
                    "spread": list(importance['spread'].items())[:5],
                    "total": list(importance['total'].items())[:5],
                }
            }
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error during XGBoost training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/retrain")
async def trigger_ml_retraining(
    min_samples: int = Query(100, description="Minimum training samples required")
):
    """Trigger manual ML model retraining (PyTorch - legacy)."""
    try:
        from ml.feedback_collector import FeedbackCollector
        from ml.trainer import CBBModelTrainer
        from ml.version_manager import ModelVersionManager

        session = SessionLocal()
        try:
            # Get training data
            collector = FeedbackCollector(session)
            df = collector.get_training_dataframe()

            if len(df) < min_samples:
                return {
                    "status": "skipped",
                    "reason": "insufficient_data",
                    "available_samples": len(df),
                    "required_samples": min_samples
                }

            # Split data (time-based)
            df = df.sort_values('game_date')
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]

            logger.info(f"Training on {len(train_df)} games, validating on {len(val_df)} games")

            # Train new model
            trainer = CBBModelTrainer(model_dir="./models")
            metrics = trainer.train(
                train_df=train_df,
                val_df=val_df,
                epochs=100,
                patience=15
            )

            # Calibrate temperature
            trainer.calibrate_temperature(val_df)

            # Save and register version
            version_str = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_pytorch"
            model_path = trainer.save_model(version_str)

            version_manager = ModelVersionManager("./models", session)
            version = version_manager.create_version(
                model_type='PyTorch',
                features=trainer.FEATURE_COLUMNS,
                hyperparameters={
                    'hidden_sizes': (64, 128, 64),
                    'dropout': 0.3,
                    'height_weight': trainer.height_weight,
                },
                training_metrics=metrics,
                model_path=model_path
            )

            # Check if new model is better and activate if so
            active_version = version_manager.get_active_version()
            activated = False
            if active_version:
                if metrics['brier_score'] < (active_version.brier_score or 1.0) - 0.005:
                    version_manager.activate_version(version_str)
                    activated = True
            else:
                version_manager.activate_version(version_str)
                activated = True

            return {
                "status": "success",
                "version": version_str,
                "activated": activated,
                "metrics": metrics,
                "training_samples": len(train_df),
                "validation_samples": len(val_df)
            }
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error during ML retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/calibration", response_model=MLCalibrationResponse)
async def get_ml_calibration():
    """Get calibration metrics for the active ML model."""
    try:
        from ml.version_manager import ModelVersionManager
        from ml.feedback_collector import FeedbackCollector

        session = SessionLocal()
        try:
            version_manager = ModelVersionManager(model_dir="./models", session=session)
            active = version_manager.get_active_version()

            if not active:
                return MLCalibrationResponse(
                    model_version=None,
                    temperature=None,
                    calibration_buckets=[],
                    overall_calibration_error=None,
                    brier_score=None
                )

            # Get calibration stats from feedback collector
            collector = FeedbackCollector(session)
            accuracy_stats = collector.get_accuracy_stats()

            # Build calibration buckets
            buckets = []
            if 'calibration_buckets' in accuracy_stats:
                buckets = accuracy_stats['calibration_buckets']

            return MLCalibrationResponse(
                model_version=active.version,
                temperature=1.0,  # Would need to load from model
                calibration_buckets=buckets,
                overall_calibration_error=active.calibration_error,
                brier_score=active.brier_score
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error fetching ML calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/feedback-stats", response_model=MLFeedbackStatsResponse)
async def get_ml_feedback_stats():
    """Get feedback loop statistics."""
    try:
        from ml.feedback_collector import FeedbackCollector
        from models.database import TrainingData
        from datetime import timedelta

        session = SessionLocal()
        try:
            collector = FeedbackCollector(session)

            # Get total samples
            total_samples = session.query(TrainingData).count()

            # Get samples from last 7 days
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            samples_7d = session.query(TrainingData).filter(
                TrainingData.created_at >= week_ago
            ).count()

            # Get samples from last 30 days
            month_ago = datetime.now(timezone.utc) - timedelta(days=30)
            samples_30d = session.query(TrainingData).filter(
                TrainingData.created_at >= month_ago
            ).count()

            # Get accuracy stats
            accuracy_stats = collector.get_accuracy_stats()

            # Calculate average errors
            avg_errors = {}
            if total_samples > 0:
                from sqlalchemy import func
                error_stats = session.query(
                    func.avg(TrainingData.win_error).label('avg_win_error'),
                    func.avg(TrainingData.spread_error).label('avg_spread_error'),
                    func.avg(TrainingData.total_error).label('avg_total_error')
                ).first()

                avg_errors = {
                    'win_error': float(error_stats.avg_win_error or 0),
                    'spread_error': float(error_stats.avg_spread_error or 0),
                    'total_error': float(error_stats.avg_total_error or 0)
                }

            return MLFeedbackStatsResponse(
                total_training_samples=total_samples,
                samples_last_7_days=samples_7d,
                samples_last_30_days=samples_30d,
                accuracy_stats=accuracy_stats,
                avg_errors=avg_errors
            )
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error fetching ML feedback stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/collect-feedback")
async def collect_feedback(
    days_back: int = Query(1, description="Number of days to look back")
):
    """Manually trigger feedback collection for completed games."""
    try:
        from ml.feedback_collector import FeedbackCollector

        session = SessionLocal()
        try:
            collector = FeedbackCollector(session)
            result = collector.run_collection(days_back=days_back, store=True)

            return {
                "status": "success",
                "games_processed": result.get('games_processed', 0),
                "new_training_samples": result.get('new_samples', 0),
                "message": f"Collected feedback for {result.get('games_processed', 0)} games"
            }
        finally:
            session.close()
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")


@app.post("/api/ml/build-training-data")
async def build_training_data_from_kenpom(
    days_back: int = Query(60, description="Number of days to look back for historical data")
):
    """
    Build training data from KenPom historical predictions and results.
    This fetches fanmatch data for past games and creates training samples.
    """
    try:
        from ml.historical_data_builder import build_training_data

        result = await build_training_data(days_back=days_back)

        if result["status"] == "success":
            # Now train the XGBoost model with this data
            from ml.xgboost_model import XGBoostPredictor, FEATURE_COLUMNS
            import pandas as pd

            df = pd.read_csv(result["output_path"])

            if len(df) < 50:
                return {
                    "status": "insufficient_data",
                    "samples_collected": len(df),
                    "message": f"Collected {len(df)} samples, need at least 50 for training"
                }

            # Split data (time-based)
            df = df.sort_values('game_date')
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]

            logger.info(f"Training XGBoost on {len(train_df)} games, validating on {len(val_df)} games")

            # Train XGBoost model
            predictor = XGBoostPredictor(model_dir="./ml_models")
            metrics = predictor.train(train_df, val_df)

            # Save model
            version_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            model_path = predictor.save(version_str)

            return {
                "status": "success",
                "samples_collected": len(df),
                "training_samples": len(train_df),
                "validation_samples": len(val_df),
                "data_path": result["output_path"],
                "model_version": f"xgboost_v{version_str}",
                "model_path": model_path,
                "metrics": metrics,
                "date_range": result.get("date_range")
            }

        return result

    except Exception as e:
        logger.error(f"Error building training data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/scheduler/status")
async def get_scheduler_status():
    """Get the status of the ML retraining scheduler."""
    try:
        # Check if scheduler is running (would need global scheduler instance)
        # For now, return configuration info
        return {
            "feedback_collection": {
                "schedule": "Daily at 11:59 PM",
                "timezone": "America/New_York"
            },
            "model_retraining": {
                "schedule": "Weekly on Sunday at 2:00 AM",
                "timezone": "America/New_York"
            },
            "status": "Scheduler not started in serverless mode" if IS_SERVERLESS else "Check server logs"
        }
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/bet-history/fetch-results")
async def fetch_and_update_results():
    """
    Fetch completed game scores and update bet results automatically.
    This endpoint checks for games from the past few days that have completed.
    """
    try:
        # Get pending bets
        pending = bet_history_service.get_pending_bets()

        if not pending:
            return {"message": "No pending bets to update", "updated": 0}

        # Try to get scores from Odds API (completed events)
        odds_client = OddsAPIClient()

        try:
            scores_data = await odds_client.get_scores(days_from=3)
        except Exception as e:
            return {"error": f"Could not fetch scores: {str(e)}", "updated": 0}

        # Build scores dict with multiple key formats for matching
        # CRITICAL: Include game_date to prevent matching wrong games
        scores = {}
        for game in scores_data:
            if game.get("completed"):
                home_team = game.get("home_team")
                away_team = game.get("away_team")

                # Extract game date from commence_time
                commence_time = game.get("commence_time", "")
                game_date = None
                if commence_time:
                    try:
                        game_dt = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                        game_date = game_dt.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse commence_time: {commence_time}")

                home_score = None
                away_score = None

                for score in game.get("scores", []):
                    if score.get("name") == home_team:
                        home_score = int(score.get("score", 0))
                    elif score.get("name") == away_team:
                        away_score = int(score.get("score", 0))

                # Validate scores - must be positive for a completed basketball game
                if home_score is not None and away_score is not None and home_score > 0 and away_score > 0:
                    score_data = {"home": home_score, "away": away_score, "game_date": game_date}
                    key = f"{home_team} vs {away_team}"
                    scores[key] = score_data
                    scores[_normalize_team_key(home_team, away_team)] = score_data

        # Track actual unique games for accurate counting
        unique_games = set()
        for key in scores.keys():
            # Only count exact keys (not normalized) to avoid double counting
            if " vs " in key and key[0].isupper():  # Exact keys start with capital letter
                unique_games.add(key)

        # Match pending bets to scores using fuzzy matching
        updated = 0
        skipped_date_mismatch = 0
        for bet in pending:
            if bet.result is not None:
                continue

            score = None
            game_key = f"{bet.home_team} vs {bet.away_team}"
            if game_key in scores:
                score = scores[game_key]
            else:
                normalized_key = _normalize_team_key(bet.home_team, bet.away_team)
                if normalized_key in scores:
                    score = scores[normalized_key]
                else:
                    score = _find_matching_score(bet.home_team, bet.away_team, scores)

            if not score:
                continue

            # CRITICAL: Validate game date matches bet date
            # This prevents matching today's bet to yesterday's game with same teams
            score_game_date = score.get("game_date")
            if score_game_date and bet.date:
                if score_game_date != bet.date:
                    logger.warning(
                        f"Skipping score match for {bet.home_team} vs {bet.away_team}: "
                        f"score date {score_game_date} != bet date {bet.date}"
                    )
                    skipped_date_mismatch += 1
                    continue

            updated += bet_history_service.update_single_bet_result(
                bet.bet_id,
                score["home"],
                score["away"]
            )

        return {
            "games_found": len(unique_games),
            "bets_updated": updated,
            "pending_remaining": len(pending) - updated,
            "message": f"Found {len(unique_games)} completed games, updated {updated} bets"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RESPONSE CONVERSION HELPERS ====================

def _stat_comparison_to_response(stat_comparison) -> Optional[TeamStatComparisonResponse]:
    """Convert TeamStatComparison to response model."""
    if not stat_comparison:
        return None

    def stats_to_response(stats) -> TeamStatsResponse:
        return TeamStatsResponse(
            team_name=stats.team_name,
            rank=stats.rank,
            adj_em=stats.adj_em,
            adj_oe=stats.adj_oe,
            adj_de=stats.adj_de,
            adj_tempo=stats.adj_tempo,
            efg_pct=stats.efg_pct,
            to_pct=stats.to_pct,
            or_pct=stats.or_pct,
            ft_rate=stats.ft_rate,
            d_efg_pct=stats.d_efg_pct,
            d_to_pct=stats.d_to_pct,
            d_or_pct=stats.d_or_pct,
            d_ft_rate=stats.d_ft_rate,
            sos=stats.sos,
            luck=stats.luck
        )

    def diff_to_response(diff) -> StatDifferenceResponse:
        return StatDifferenceResponse(
            stat_name=diff.stat_name,
            display_name=diff.display_name,
            home_value=diff.home_value,
            away_value=diff.away_value,
            difference=diff.difference,
            advantage=diff.advantage,
            significance=diff.significance,
            higher_is_better=diff.higher_is_better
        )

    return TeamStatComparisonResponse(
        home_stats=stats_to_response(stat_comparison.home_stats),
        away_stats=stats_to_response(stat_comparison.away_stats),
        stat_differences=[diff_to_response(d) for d in stat_comparison.stat_differences],
        major_differences=[diff_to_response(d) for d in stat_comparison.major_differences],
        efficiency_edge=stat_comparison.efficiency_edge,
        tempo_mismatch=stat_comparison.tempo_mismatch,
        shooting_edge=stat_comparison.shooting_edge,
        rebounding_edge=stat_comparison.rebounding_edge,
        turnover_edge=stat_comparison.turnover_edge
    )


def _analysis_to_response(analysis) -> GameAnalysisResponse:
    """Convert GameAnalysis to response model."""
    stat_comparison_response = _stat_comparison_to_response(analysis.stat_comparison)

    # Build major stat diffs from the analysis
    major_diffs = []
    if analysis.major_stat_diffs:
        for diff in analysis.major_stat_diffs:
            major_diffs.append(StatDifferenceResponse(
                stat_name=diff.stat_name,
                display_name=diff.display_name,
                home_value=diff.home_value,
                away_value=diff.away_value,
                difference=diff.difference,
                advantage=diff.advantage,
                significance=diff.significance,
                higher_is_better=diff.higher_is_better
            ))

    return GameAnalysisResponse(
        home_team=analysis.home_team,
        away_team=analysis.away_team,
        game_time=analysis.game_time,
        home_rank=analysis.home_rank,
        away_rank=analysis.away_rank,
        kenpom_home_score=analysis.kenpom_home_score,
        kenpom_away_score=analysis.kenpom_away_score,
        kenpom_spread=analysis.kenpom_spread,
        kenpom_total=analysis.kenpom_total,
        kenpom_home_win_prob=analysis.kenpom_home_win_prob,
        kenpom_tempo=analysis.kenpom_tempo,
        vegas_spread=analysis.vegas_spread,
        vegas_total=analysis.vegas_total,
        vegas_ml_home=analysis.vegas_ml_home,
        vegas_ml_away=analysis.vegas_ml_away,
        spread_diff=analysis.spread_diff,
        total_diff=analysis.total_diff,
        value_bets=[
            ValueBetResponse(
                home_team=analysis.home_team,
                away_team=analysis.away_team,
                bet_type=vb.bet_type.value,
                side=vb.side,
                model_prob=vb.model_prob,
                market_odds=vb.market_odds,
                market_line=vb.market_line,
                model_line=vb.model_line,
                market_implied_prob=vb.market_implied_prob,
                edge=vb.edge,
                ev=vb.ev,
                kelly_fraction=vb.kelly_fraction,
                recommended_book=vb.recommended_book,
                confidence=vb.confidence
            )
            for vb in analysis.value_bets
        ],
        stat_comparison=stat_comparison_response,
        major_stat_diffs=major_diffs,
        data_warnings=[
            DataQualityWarningResponse(
                code=w.code,
                severity=w.severity,
                message=w.message,
                details=w.details
            )
            for w in (analysis.data_warnings or [])
        ]
    )


# Vercel serverless handler
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    handler = None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
