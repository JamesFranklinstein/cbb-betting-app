"""
KenPom API Client

Handles all interactions with the KenPom API for college basketball analytics.
Based on official API documentation.
"""

import httpx
import logging
from typing import Optional, List, Dict, Any
from datetime import date, datetime, timezone, timedelta
import os
from dotenv import load_dotenv


def _get_us_eastern_date() -> date:
    """Get the current date in US Eastern timezone."""
    # US Eastern is UTC-5 (or UTC-4 during DST)
    # For simplicity, we'll use UTC-5 which is safe for late night UTC times
    utc_now = datetime.now(timezone.utc)
    eastern_offset = timedelta(hours=-5)
    eastern_now = utc_now + eastern_offset
    return eastern_now.date()

load_dotenv()

logger = logging.getLogger(__name__)


class KenPomClientError(Exception):
    """Custom exception for KenPom client errors."""
    pass


class KenPomClient:
    """Client for interacting with the KenPom API."""

    BASE_URL = "https://kenpom.com/api.php"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("KENPOM_API_KEY")
        if not self.api_key:
            logger.error("KenPom API key not found in environment or parameters")
            raise KenPomClientError(
                "KenPom API key is required. Set KENPOM_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
    
    async def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an authenticated request to the KenPom API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.BASE_URL,
                params=params,
                headers=self.headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    
    # ==================== RATINGS ====================
    
    async def get_ratings(
        self,
        year: Optional[int] = None,
        team_id: Optional[int] = None,
        conference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve team ratings, strength of schedule, tempo, and possession length data.

        Args:
            year: Season year (e.g., 2025 for 2024-25 season). Defaults to current season.
            team_id: Specific team ID
            conference: Conference short name (e.g., 'B12', 'SEC')

        Returns:
            List of team rating data
        """
        params = {"endpoint": "ratings"}

        # Year is required by the API - default to current season if not specified
        if year:
            params["y"] = year
        else:
            # Determine current season year (season year is the ending year, e.g., 2025-26 season = 2026)
            from datetime import date
            today = date.today()
            # If we're before July, we're in the season that ends this year
            # If we're in July or later, we're in the season that ends next year
            current_season = today.year if today.month < 7 else today.year + 1
            params["y"] = current_season
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference

        return await self._request(params)
    
    async def get_team_rating(self, team_id: int, year: Optional[int] = None) -> Dict[str, Any]:
        """Get rating data for a specific team."""
        ratings = await self.get_ratings(team_id=team_id, year=year)
        return ratings[0] if ratings else {}
    
    # ==================== RATINGS ARCHIVE ====================
    
    async def get_archive(
        self,
        date_str: Optional[str] = None,
        year: Optional[int] = None,
        preseason: bool = False,
        team_id: Optional[int] = None,
        conference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical team ratings from specific dates.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            year: Season year (required if preseason=True)
            preseason: If True, get preseason ratings
            team_id: Specific team ID
            conference: Conference short name
            
        Returns:
            List of archived rating data
        """
        params = {"endpoint": "archive"}
        
        if preseason:
            params["preseason"] = "true"
            if year:
                params["y"] = year
        elif date_str:
            params["d"] = date_str
            
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
            
        return await self._request(params)
    
    # ==================== FOUR FACTORS ====================
    
    async def get_four_factors(
        self,
        year: Optional[int] = None,
        team_id: Optional[int] = None,
        conference: Optional[str] = None,
        conf_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve Four Factors statistics (eFG%, TO%, OR%, FT Rate) for offense and defense.

        Args:
            year: Season year. Defaults to current season.
            team_id: Specific team ID
            conference: Conference short name
            conf_only: If True, return conference-only stats

        Returns:
            List of four factors data
        """
        params = {"endpoint": "four-factors"}

        # Year is required by the API - default to current season if not specified
        if year:
            params["y"] = year
        else:
            from datetime import date
            today = date.today()
            current_season = today.year if today.month < 7 else today.year + 1
            params["y"] = current_season
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        if conf_only:
            params["conf_only"] = "true"

        return await self._request(params)
    
    # ==================== POINT DISTRIBUTION ====================
    
    async def get_point_distribution(
        self,
        year: Optional[int] = None,
        team_id: Optional[int] = None,
        conference: Optional[str] = None,
        conf_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve percentage of points from FT, 2PT, and 3PT for offense and defense.
        
        Args:
            year: Season year
            team_id: Specific team ID
            conference: Conference short name
            conf_only: If True, return conference-only stats
            
        Returns:
            List of point distribution data
        """
        params = {"endpoint": "pointdist"}
        
        if year:
            params["y"] = year
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        if conf_only:
            params["conf_only"] = "true"
            
        return await self._request(params)
    
    # ==================== HEIGHT ====================
    
    async def get_height(
        self,
        year: Optional[int] = None,
        team_id: Optional[int] = None,
        conference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve team height, experience, bench strength, and continuity stats.
        
        Args:
            year: Season year
            team_id: Specific team ID
            conference: Conference short name
            
        Returns:
            List of height/experience data
        """
        params = {"endpoint": "height"}
        
        if year:
            params["y"] = year
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
            
        return await self._request(params)
    
    # ==================== MISC STATS ====================
    
    async def get_misc_stats(
        self,
        year: Optional[int] = None,
        team_id: Optional[int] = None,
        conference: Optional[str] = None,
        conf_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve miscellaneous stats (shooting %, blocks, steals, assists, etc.).
        
        Args:
            year: Season year
            team_id: Specific team ID
            conference: Conference short name
            conf_only: If True, return conference-only stats
            
        Returns:
            List of misc stats data
        """
        params = {"endpoint": "misc-stats"}
        
        if year:
            params["y"] = year
        if team_id:
            params["team_id"] = team_id
        if conference:
            params["c"] = conference
        if conf_only:
            params["conf_only"] = "true"
            
        return await self._request(params)
    
    # ==================== FANMATCH (PREDICTIONS) ====================
    
    async def get_fanmatch(self, game_date: str) -> List[Dict[str, Any]]:
        """
        Retrieve game predictions for a specific date.
        
        Args:
            game_date: Date in YYYY-MM-DD format
            
        Returns:
            List of game predictions with scores, win probabilities, tempo
        """
        params = {
            "endpoint": "fanmatch",
            "d": game_date
        }
        return await self._request(params)
    
    async def get_todays_predictions(self) -> List[Dict[str, Any]]:
        """Get predictions for today's games (using US Eastern timezone).

        Falls back to yesterday's date if today's data is not yet available.
        """
        today = _get_us_eastern_date()
        today_str = today.isoformat()

        try:
            result = await self.get_fanmatch(today_str)
            if result:  # If we got data, return it
                return result
        except Exception as e:
            logger.warning(f"Failed to get predictions for today ({today_str}): {e}")

        # Fallback to yesterday if today's data is empty or failed
        yesterday = today - timedelta(days=1)
        yesterday_str = yesterday.isoformat()
        logger.info(f"Falling back to yesterday's predictions ({yesterday_str})")

        try:
            return await self.get_fanmatch(yesterday_str)
        except Exception as e:
            logger.error(f"Failed to get predictions for yesterday ({yesterday_str}): {e}")
            raise KenPomClientError(f"No predictions available for today or yesterday")
    
    # ==================== CONFERENCE RATINGS ====================
    
    async def get_conference_ratings(
        self,
        year: Optional[int] = None,
        conference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conference ratings.
        
        Args:
            year: Season year
            conference: Conference short name
            
        Returns:
            List of conference rating data
        """
        params = {"endpoint": "conf-ratings"}
        
        if year:
            params["y"] = year
        if conference:
            params["c"] = conference
            
        return await self._request(params)
    
    # ==================== TEAMS ====================
    
    async def get_teams(
        self,
        year: int,
        conference: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve list of teams for a season.
        
        Args:
            year: Season year (required)
            conference: Conference short name
            
        Returns:
            List of teams with IDs, names, conferences, coaches
        """
        params = {
            "endpoint": "teams",
            "y": year
        }
        
        if conference:
            params["c"] = conference
            
        return await self._request(params)
    
    async def get_team_id(self, team_name: str, year: int) -> Optional[int]:
        """Look up a team ID by name."""
        teams = await self.get_teams(year)
        for team in teams:
            if team_name.lower() in team["TeamName"].lower():
                return team["TeamID"]
        return None
    
    # ==================== CONFERENCES ====================
    
    async def get_conferences(self, year: int) -> List[Dict[str, Any]]:
        """
        Retrieve list of conferences for a season.
        
        Args:
            year: Season year (required)
            
        Returns:
            List of conferences with IDs and names
        """
        params = {
            "endpoint": "conferences",
            "y": year
        }
        return await self._request(params)


# Synchronous wrapper for CLI usage
def get_sync_client() -> KenPomClient:
    """Get a KenPom client for synchronous usage."""
    return KenPomClient()
