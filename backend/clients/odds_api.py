"""
The Odds API Client

Handles all interactions with The Odds API for sports betting odds.
Supports NCAAB (college basketball) odds from multiple bookmakers.
"""

import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
from dotenv import load_dotenv
from enum import Enum

load_dotenv()


class Market(Enum):
    """Available betting markets."""
    H2H = "h2h"           # Moneyline
    SPREADS = "spreads"   # Point spread
    TOTALS = "totals"     # Over/Under


class OddsFormat(Enum):
    """Odds display format."""
    AMERICAN = "american"  # +150, -200
    DECIMAL = "decimal"    # 2.50, 1.50


class OddsAPIClient:
    """Client for interacting with The Odds API."""
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT_KEY = "basketball_ncaab"  # College basketball
    
    # Popular US bookmakers
    DEFAULT_BOOKMAKERS = [
        "draftkings",
        "fanduel",
        "betmgm",
        "caesars",
        "pointsbet",
        "bet365",
        "bovada"
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            raise ValueError("Odds API key is required")
        
        self.remaining_requests = None
        self.used_requests = None
    
    def _update_usage(self, headers: dict):
        """Track API usage from response headers."""
        self.remaining_requests = headers.get("x-requests-remaining")
        self.used_requests = headers.get("x-requests-used")
    
    async def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make an authenticated request to The Odds API."""
        params["apiKey"] = self.api_key
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.BASE_URL}/{endpoint}",
                params=params,
                timeout=30.0
            )
            self._update_usage(response.headers)
            response.raise_for_status()
            return response.json()
    
    # ==================== SPORTS ====================
    
    async def get_sports(self, all_sports: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available sports.
        
        Args:
            all_sports: If True, include out-of-season sports
            
        Returns:
            List of sports with keys and titles
        """
        params = {}
        if all_sports:
            params["all"] = "true"
        
        return await self._request("sports", params)
    
    # ==================== ODDS ====================
    
    async def get_odds(
        self,
        markets: List[Market] = None,
        regions: List[str] = None,
        bookmakers: List[str] = None,
        odds_format: OddsFormat = OddsFormat.AMERICAN,
        event_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get current odds for NCAAB games.
        
        Args:
            markets: List of betting markets (h2h, spreads, totals)
            regions: List of regions (us, uk, eu, au)
            bookmakers: Specific bookmakers to include
            odds_format: American or decimal odds format
            event_ids: Specific event IDs to fetch
            
        Returns:
            List of games with odds from bookmakers
        """
        if markets is None:
            markets = [Market.H2H, Market.SPREADS, Market.TOTALS]
        
        if regions is None:
            regions = ["us"]
        
        params = {
            "regions": ",".join(regions),
            "markets": ",".join(m.value for m in markets),
            "oddsFormat": odds_format.value
        }
        
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        
        if event_ids:
            params["eventIds"] = ",".join(event_ids)
        
        return await self._request(f"sports/{self.SPORT_KEY}/odds", params)
    
    async def get_all_odds(self) -> List[Dict[str, Any]]:
        """Get all available odds for today's NCAAB games."""
        return await self.get_odds(
            markets=[Market.H2H, Market.SPREADS, Market.TOTALS],
            regions=["us"],
            odds_format=OddsFormat.AMERICAN
        )
    
    # ==================== EVENTS ====================
    
    async def get_events(self, event_ids: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of upcoming NCAAB events.
        
        Args:
            event_ids: Optional specific event IDs
            
        Returns:
            List of upcoming games
        """
        params = {}
        if event_ids:
            params["eventIds"] = ",".join(event_ids)
        
        return await self._request(f"sports/{self.SPORT_KEY}/events", params)
    
    # ==================== SCORES ====================
    
    async def get_scores(self, days_from: int = 1) -> List[Dict[str, Any]]:
        """
        Get scores for recent NCAAB games.
        
        Args:
            days_from: Number of days in the past to include (1-3)
            
        Returns:
            List of games with scores
        """
        params = {"daysFrom": min(days_from, 3)}
        return await self._request(f"sports/{self.SPORT_KEY}/scores", params)
    
    # ==================== HELPER METHODS ====================
    
    def extract_best_odds(
        self,
        game: Dict[str, Any],
        market: Market,
        team: str
    ) -> Dict[str, Any]:
        """
        Find the best odds for a specific team and market across all bookmakers.
        
        Args:
            game: Game data from get_odds()
            market: The betting market
            team: Team name to find odds for
            
        Returns:
            Dict with best odds, bookmaker, and line (if applicable)
        """
        best_odds = None
        best_bookmaker = None
        best_point = None
        
        for bookmaker in game.get("bookmakers", []):
            for mkt in bookmaker.get("markets", []):
                if mkt["key"] != market.value:
                    continue
                
                for outcome in mkt.get("outcomes", []):
                    if outcome["name"] == team:
                        price = outcome["price"]
                        point = outcome.get("point")
                        
                        # For American odds, higher is better for positive
                        # and higher (less negative) is better for negative
                        if best_odds is None or price > best_odds:
                            best_odds = price
                            best_bookmaker = bookmaker["title"]
                            best_point = point
        
        return {
            "odds": best_odds,
            "bookmaker": best_bookmaker,
            "point": best_point
        }
    
    def get_consensus_line(
        self,
        game: Dict[str, Any],
        market: Market
    ) -> Dict[str, Any]:
        """
        Calculate the consensus (average) line across bookmakers.
        
        Args:
            game: Game data from get_odds()
            market: The betting market
            
        Returns:
            Dict with average line/odds for each team
        """
        home_team = game["home_team"]
        away_team = game["away_team"]
        
        home_prices = []
        away_prices = []
        home_points = []
        away_points = []
        
        for bookmaker in game.get("bookmakers", []):
            for mkt in bookmaker.get("markets", []):
                if mkt["key"] != market.value:
                    continue
                
                for outcome in mkt.get("outcomes", []):
                    if outcome["name"] == home_team:
                        home_prices.append(outcome["price"])
                        if "point" in outcome:
                            home_points.append(outcome["point"])
                    elif outcome["name"] == away_team:
                        away_prices.append(outcome["price"])
                        if "point" in outcome:
                            away_points.append(outcome["point"])
        
        result = {
            "home_team": home_team,
            "away_team": away_team,
            "home_avg_odds": sum(home_prices) / len(home_prices) if home_prices else None,
            "away_avg_odds": sum(away_prices) / len(away_prices) if away_prices else None,
        }
        
        if home_points:
            result["home_avg_point"] = sum(home_points) / len(home_points)
        if away_points:
            result["away_avg_point"] = sum(away_points) / len(away_points)
        
        return result
    
    def american_to_implied_prob(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def implied_prob_to_american(self, prob: float) -> int:
        """Convert implied probability to American odds."""
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        else:
            return int(100 * (1 - prob) / prob)
    
    def get_api_usage(self) -> Dict[str, Any]:
        """Get current API usage statistics."""
        return {
            "remaining_requests": self.remaining_requests,
            "used_requests": self.used_requests
        }


# Factory function
def get_odds_client() -> OddsAPIClient:
    """Get an Odds API client instance."""
    return OddsAPIClient()
