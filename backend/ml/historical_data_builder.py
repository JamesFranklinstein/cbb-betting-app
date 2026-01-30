"""
Historical Data Builder

Builds training data from KenPom historical predictions and results.
Uses the archive endpoint to get historical ratings and fanmatch for predictions.
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from clients.kenpom import KenPomClient

logger = logging.getLogger(__name__)


# Feature columns for ML model
FEATURE_COLUMNS = [
    "adj_em_diff", "adj_oe_diff", "adj_de_diff", "adj_tempo_diff",
    "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
    "d_efg_pct_diff", "d_to_pct_diff", "d_or_pct_diff", "d_ft_rate_diff",
    "sos_diff", "luck_diff", "home_advantage", "rank_diff",
    "home_win_streak", "away_win_streak",
    "height_diff", "effective_height_diff", "height_vs_tempo",
]


class HistoricalDataBuilder:
    """Builds training data from KenPom historical data."""

    def __init__(self):
        self.kenpom = KenPomClient()
        self._ratings_cache: Dict[str, Dict[str, Any]] = {}
        self._four_factors_cache: Dict[str, Dict[str, Any]] = {}

    async def _load_ratings_for_date(self, date_str: str) -> None:
        """Load ratings and four factors for a specific date into cache."""
        try:
            # Get ratings from archive
            ratings = await self.kenpom.get_archive(date_str=date_str)
            four_factors = await self.kenpom.get_four_factors()

            # Build lookup by team name
            self._ratings_cache = {}
            for team in ratings:
                name = team.get("TeamName", "").lower()
                if name:
                    self._ratings_cache[name] = team

            self._four_factors_cache = {}
            for team in four_factors:
                name = team.get("TeamName", "").lower()
                if name:
                    self._four_factors_cache[name] = team

            logger.info(f"Loaded {len(self._ratings_cache)} teams for {date_str}")

        except Exception as e:
            logger.warning(f"Could not load ratings for {date_str}: {e}")

    def _get_team_data(self, team_name: str) -> Dict[str, float]:
        """Get team stats from cache."""
        normalized = team_name.lower()

        # Try exact match first
        ratings = self._ratings_cache.get(normalized, {})
        four_factors = self._four_factors_cache.get(normalized, {})

        # Try partial match
        if not ratings:
            for cached_name, data in self._ratings_cache.items():
                if normalized in cached_name or cached_name in normalized:
                    ratings = data
                    break

        if not four_factors:
            for cached_name, data in self._four_factors_cache.items():
                if normalized in cached_name or cached_name in normalized:
                    four_factors = data
                    break

        return {
            "adj_em": ratings.get("AdjEM", 0.0),
            "adj_oe": ratings.get("AdjOE", 100.0),
            "adj_de": ratings.get("AdjDE", 100.0),
            "adj_tempo": ratings.get("AdjTempo", 68.0),
            "rank": ratings.get("RankAdjEM", 200),
            "sos": ratings.get("SOS", 0.0),
            "luck": ratings.get("Luck", 0.0),
            "efg_pct": four_factors.get("eFG_Pct", 50.0),
            "to_pct": four_factors.get("TO_Pct", 18.0),
            "or_pct": four_factors.get("OR_Pct", 30.0),
            "ft_rate": four_factors.get("FT_Rate", 30.0),
            "d_efg_pct": four_factors.get("DeFG_Pct", 50.0),
            "d_to_pct": four_factors.get("DTO_Pct", 18.0),
            "d_or_pct": four_factors.get("DOR_Pct", 30.0),
            "d_ft_rate": four_factors.get("DFT_Rate", 30.0),
        }

    def _compute_features(
        self,
        home_stats: Dict[str, float],
        away_stats: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute feature differences between home and away teams."""
        features = {
            "adj_em_diff": home_stats["adj_em"] - away_stats["adj_em"],
            "adj_oe_diff": home_stats["adj_oe"] - away_stats["adj_oe"],
            "adj_de_diff": away_stats["adj_de"] - home_stats["adj_de"],  # Lower is better for defense
            "adj_tempo_diff": home_stats["adj_tempo"] - away_stats["adj_tempo"],
            "efg_pct_diff": home_stats["efg_pct"] - away_stats["efg_pct"],
            "to_pct_diff": away_stats["to_pct"] - home_stats["to_pct"],  # Lower is better
            "or_pct_diff": home_stats["or_pct"] - away_stats["or_pct"],
            "ft_rate_diff": home_stats["ft_rate"] - away_stats["ft_rate"],
            "d_efg_pct_diff": away_stats["d_efg_pct"] - home_stats["d_efg_pct"],  # Lower is better
            "d_to_pct_diff": home_stats["d_to_pct"] - away_stats["d_to_pct"],
            "d_or_pct_diff": away_stats["d_or_pct"] - home_stats["d_or_pct"],  # Lower is better
            "d_ft_rate_diff": away_stats["d_ft_rate"] - home_stats["d_ft_rate"],  # Lower is better
            "sos_diff": home_stats["sos"] - away_stats["sos"],
            "luck_diff": home_stats["luck"] - away_stats["luck"],
            "home_advantage": 3.5,  # Standard home court advantage
            "rank_diff": away_stats["rank"] - home_stats["rank"],  # Positive = home is better ranked
            "home_win_streak": 0.0,  # Not available from KenPom
            "away_win_streak": 0.0,
            "height_diff": 0.0,  # Would need height data
            "effective_height_diff": 0.0,
            "height_vs_tempo": 0.0,
        }
        return features

    async def build_from_fanmatch_history(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Build training data from KenPom fanmatch (predictions) for a date range.

        Note: This requires that games have actually been played (fanmatch shows results).

        Args:
            start_date: Start date for data collection
            end_date: End date for data collection

        Returns:
            DataFrame with training data
        """
        all_data = []
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.isoformat()
            logger.info(f"Processing {date_str}...")

            try:
                # Load ratings for this date
                await self._load_ratings_for_date(date_str)

                # Get fanmatch data (predictions and results if game completed)
                games = await self.kenpom.get_fanmatch(date_str)

                for game in games:
                    try:
                        home_team = game.get("Home", "")
                        away_team = game.get("Visitor", "")

                        # Get predicted scores
                        pred_home = game.get("HomePred", 0)
                        pred_away = game.get("VisitorPred", 0)

                        # Check if we have actual results (game completed)
                        # KenPom fanmatch shows actual scores after game completes
                        actual_home = game.get("HomeScore")
                        actual_away = game.get("VisitorScore")

                        # Skip if no actual results
                        if actual_home is None or actual_away is None:
                            continue

                        # Get team stats
                        home_stats = self._get_team_data(home_team)
                        away_stats = self._get_team_data(away_team)

                        # Compute features
                        features = self._compute_features(home_stats, away_stats)

                        # Create training sample
                        sample = {
                            "game_date": date_str,
                            "home_team": home_team,
                            "away_team": away_team,
                            "home_rank": game.get("HomeRank", 0),
                            "away_rank": game.get("VisitorRank", 0),
                            "kenpom_home_score": pred_home,
                            "kenpom_away_score": pred_away,
                            "kenpom_spread": pred_away - pred_home,
                            "kenpom_total": pred_home + pred_away,
                            "kenpom_home_win_prob": game.get("HomeWP", 0.5),
                            "actual_home_score": actual_home,
                            "actual_away_score": actual_away,
                            "actual_spread": actual_home - actual_away,
                            "actual_total": actual_home + actual_away,
                            "home_won": 1 if actual_home > actual_away else 0,
                            **features
                        }

                        all_data.append(sample)

                    except Exception as e:
                        logger.warning(f"Error processing game {home_team} vs {away_team}: {e}")
                        continue

                # Rate limit - be nice to KenPom API
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.warning(f"Error processing date {date_str}: {e}")

            current_date += timedelta(days=1)

        if not all_data:
            logger.warning("No training data collected")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        logger.info(f"Built {len(df)} training samples from {start_date} to {end_date}")

        return df

    async def build_current_season_data(self, days_back: int = 60) -> pd.DataFrame:
        """
        Build training data for the current season.

        Args:
            days_back: Number of days to look back

        Returns:
            DataFrame with training data
        """
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days_back)

        return await self.build_from_fanmatch_history(start_date, end_date)


async def build_training_data(days_back: int = 60) -> Dict[str, Any]:
    """
    Build training data from KenPom historical data.

    Args:
        days_back: Number of days to look back

    Returns:
        Dict with status and data path
    """
    builder = HistoricalDataBuilder()

    try:
        df = await builder.build_current_season_data(days_back=days_back)

        if df.empty:
            return {
                "status": "error",
                "message": "No training data could be collected",
                "samples": 0
            }

        # Save to CSV
        output_path = f"./ml_models/training_data_{date.today().isoformat()}.csv"
        df.to_csv(output_path, index=False)

        return {
            "status": "success",
            "samples": len(df),
            "output_path": output_path,
            "date_range": {
                "start": df["game_date"].min(),
                "end": df["game_date"].max()
            },
            "columns": list(df.columns)
        }

    except Exception as e:
        logger.error(f"Error building training data: {e}")
        return {
            "status": "error",
            "message": str(e),
            "samples": 0
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    days = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    result = asyncio.run(build_training_data(days_back=days))
    print(f"\nResult: {result}")
