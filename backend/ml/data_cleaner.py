"""
Data Cleaning Utility for CBB Betting App

Identifies and removes false/incorrect results from the training data.
False results include:
- Games graded before they occurred (date mismatch)
- Inconsistent scores for the same game
- Impossible scores (too low, too high, negative)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from models.database import StoredBet, TrainingData, Game
from models.connection import SessionLocal

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans and validates training data for ML models."""

    # Score validation thresholds
    MIN_VALID_SCORE = 30  # CBB games rarely have scores below 30
    MAX_VALID_SCORE = 150  # CBB games rarely have scores above 150

    def __init__(self, session: Optional[Session] = None):
        self.session = session or SessionLocal()
        self._owns_session = session is None

    def close(self):
        """Close the session if we own it."""
        if self._owns_session:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ==================== ANALYSIS ====================

    def analyze_stored_bets(self) -> Dict[str, Any]:
        """
        Analyze StoredBet table for data quality issues.

        Returns:
            Dict with analysis results including counts and problematic records
        """
        all_bets = self.session.query(StoredBet).all()

        issues = {
            "total_bets": len(all_bets),
            "graded_bets": 0,
            "pending_bets": 0,
            "invalid_scores": [],
            "inconsistent_games": [],
            "future_graded": [],
        }

        # Group bets by game (home_team, away_team, date)
        games = defaultdict(list)
        today = datetime.now().strftime("%Y-%m-%d")

        for bet in all_bets:
            if bet.result is not None:
                issues["graded_bets"] += 1

                # Check for future games that are graded
                if bet.date and bet.date >= today and bet.result is not None:
                    issues["future_graded"].append({
                        "bet_id": bet.bet_id,
                        "date": bet.date,
                        "teams": f"{bet.away_team} @ {bet.home_team}",
                        "result": bet.result,
                        "score": f"{bet.away_score}-{bet.home_score}"
                    })

                # Check for invalid scores
                if bet.home_score is not None and bet.away_score is not None:
                    if (bet.home_score < self.MIN_VALID_SCORE or
                        bet.away_score < self.MIN_VALID_SCORE or
                        bet.home_score > self.MAX_VALID_SCORE or
                        bet.away_score > self.MAX_VALID_SCORE or
                        bet.home_score < 0 or bet.away_score < 0):
                        issues["invalid_scores"].append({
                            "bet_id": bet.bet_id,
                            "date": bet.date,
                            "teams": f"{bet.away_team} @ {bet.home_team}",
                            "score": f"{bet.away_score}-{bet.home_score}"
                        })

                # Group by game for consistency check
                game_key = (bet.home_team, bet.away_team, bet.date)
                games[game_key].append(bet)
            else:
                issues["pending_bets"] += 1

        # Check for inconsistent scores within same game
        for game_key, bets in games.items():
            if len(bets) > 1:
                scores = set()
                for bet in bets:
                    if bet.home_score is not None and bet.away_score is not None:
                        scores.add((bet.home_score, bet.away_score))

                if len(scores) > 1:
                    issues["inconsistent_games"].append({
                        "game": f"{game_key[1]} @ {game_key[0]} ({game_key[2]})",
                        "scores": list(scores),
                        "bet_ids": [b.bet_id for b in bets]
                    })

        return issues

    def analyze_training_data(self) -> Dict[str, Any]:
        """
        Analyze TrainingData table for data quality issues.

        Returns:
            Dict with analysis results
        """
        all_data = self.session.query(TrainingData).all()

        issues = {
            "total_records": len(all_data),
            "missing_scores": [],
            "invalid_scores": [],
            "missing_features": [],
            "large_errors": [],
        }

        for record in all_data:
            # Check for missing actual scores
            if record.actual_home_score is None or record.actual_away_score is None:
                issues["missing_scores"].append(record.id)
            else:
                # Check for invalid scores
                if (record.actual_home_score < self.MIN_VALID_SCORE or
                    record.actual_away_score < self.MIN_VALID_SCORE or
                    record.actual_home_score > self.MAX_VALID_SCORE or
                    record.actual_away_score > self.MAX_VALID_SCORE):
                    issues["invalid_scores"].append({
                        "id": record.id,
                        "game_date": record.game_date,
                        "score": f"{record.actual_away_score}-{record.actual_home_score}"
                    })

            # Check for missing features
            if not record.features:
                issues["missing_features"].append(record.id)

            # Check for large prediction errors (might indicate wrong results)
            if record.spread_error is not None and abs(record.spread_error) > 30:
                issues["large_errors"].append({
                    "id": record.id,
                    "game_date": record.game_date,
                    "spread_error": record.spread_error,
                    "total_error": record.total_error
                })

        return issues

    # ==================== CLEANING ====================

    def reset_future_graded_bets(self, dry_run: bool = True) -> List[str]:
        """
        Reset bets that were graded but have future dates.

        Args:
            dry_run: If True, only report what would be reset

        Returns:
            List of bet_ids that were (or would be) reset
        """
        today = datetime.now().strftime("%Y-%m-%d")

        future_graded = self.session.query(StoredBet).filter(
            and_(
                StoredBet.date >= today,
                StoredBet.result.isnot(None)
            )
        ).all()

        reset_ids = []
        for bet in future_graded:
            reset_ids.append(bet.bet_id)
            if not dry_run:
                bet.result = None
                bet.home_score = None
                bet.away_score = None
                bet.actual_margin = None
                bet.actual_total = None
                bet.settled_at = None

        if not dry_run:
            self.session.commit()
            logger.info(f"Reset {len(reset_ids)} future-graded bets")

        return reset_ids

    def reset_invalid_score_bets(self, dry_run: bool = True) -> List[str]:
        """
        Reset bets with invalid scores.

        Args:
            dry_run: If True, only report what would be reset

        Returns:
            List of bet_ids that were (or would be) reset
        """
        invalid_bets = self.session.query(StoredBet).filter(
            and_(
                StoredBet.result.isnot(None),
                or_(
                    StoredBet.home_score < self.MIN_VALID_SCORE,
                    StoredBet.away_score < self.MIN_VALID_SCORE,
                    StoredBet.home_score > self.MAX_VALID_SCORE,
                    StoredBet.away_score > self.MAX_VALID_SCORE,
                    StoredBet.home_score < 0,
                    StoredBet.away_score < 0
                )
            )
        ).all()

        reset_ids = []
        for bet in invalid_bets:
            reset_ids.append(bet.bet_id)
            if not dry_run:
                bet.result = None
                bet.home_score = None
                bet.away_score = None
                bet.actual_margin = None
                bet.actual_total = None
                bet.settled_at = None

        if not dry_run:
            self.session.commit()
            logger.info(f"Reset {len(reset_ids)} invalid-score bets")

        return reset_ids

    def reset_inconsistent_game_bets(self, dry_run: bool = True) -> List[str]:
        """
        Reset all bets for games that have inconsistent scores.

        Args:
            dry_run: If True, only report what would be reset

        Returns:
            List of bet_ids that were (or would be) reset
        """
        # Find inconsistent games
        all_bets = self.session.query(StoredBet).filter(
            StoredBet.result.isnot(None)
        ).all()

        # Group by game
        games = defaultdict(list)
        for bet in all_bets:
            game_key = (bet.home_team, bet.away_team, bet.date)
            games[game_key].append(bet)

        reset_ids = []
        for game_key, bets in games.items():
            if len(bets) > 1:
                scores = set()
                for bet in bets:
                    if bet.home_score is not None and bet.away_score is not None:
                        scores.add((bet.home_score, bet.away_score))

                # If multiple different scores, reset all bets for this game
                if len(scores) > 1:
                    for bet in bets:
                        reset_ids.append(bet.bet_id)
                        if not dry_run:
                            bet.result = None
                            bet.home_score = None
                            bet.away_score = None
                            bet.actual_margin = None
                            bet.actual_total = None
                            bet.settled_at = None

        if not dry_run:
            self.session.commit()
            logger.info(f"Reset {len(reset_ids)} bets from inconsistent games")

        return reset_ids

    def delete_invalid_training_data(self, dry_run: bool = True) -> int:
        """
        Delete training data records with invalid or missing data.

        Args:
            dry_run: If True, only report what would be deleted

        Returns:
            Number of records deleted (or would be deleted)
        """
        # Find invalid records
        invalid_records = self.session.query(TrainingData).filter(
            or_(
                TrainingData.actual_home_score.is_(None),
                TrainingData.actual_away_score.is_(None),
                TrainingData.actual_home_score < self.MIN_VALID_SCORE,
                TrainingData.actual_away_score < self.MIN_VALID_SCORE,
                TrainingData.actual_home_score > self.MAX_VALID_SCORE,
                TrainingData.actual_away_score > self.MAX_VALID_SCORE,
                TrainingData.features.is_(None)
            )
        ).all()

        count = len(invalid_records)

        if not dry_run:
            for record in invalid_records:
                self.session.delete(record)
            self.session.commit()
            logger.info(f"Deleted {count} invalid training data records")

        return count

    def clean_all(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Run all cleaning operations.

        Args:
            dry_run: If True, only report what would be done

        Returns:
            Summary of all cleaning operations
        """
        results = {
            "dry_run": dry_run,
            "future_graded_reset": self.reset_future_graded_bets(dry_run),
            "invalid_scores_reset": self.reset_invalid_score_bets(dry_run),
            "inconsistent_games_reset": self.reset_inconsistent_game_bets(dry_run),
            "invalid_training_deleted": self.delete_invalid_training_data(dry_run),
        }

        results["summary"] = {
            "total_bets_reset": (
                len(results["future_graded_reset"]) +
                len(results["invalid_scores_reset"]) +
                len(results["inconsistent_games_reset"])
            ),
            "training_records_deleted": results["invalid_training_deleted"]
        }

        return results


def run_data_analysis():
    """Run data analysis and print report."""
    with DataCleaner() as cleaner:
        print("=" * 60)
        print("STORED BETS ANALYSIS")
        print("=" * 60)

        bet_issues = cleaner.analyze_stored_bets()
        print(f"Total bets: {bet_issues['total_bets']}")
        print(f"Graded: {bet_issues['graded_bets']}")
        print(f"Pending: {bet_issues['pending_bets']}")
        print(f"\nFuture-graded bets: {len(bet_issues['future_graded'])}")
        for item in bet_issues['future_graded'][:5]:
            print(f"  - {item['teams']} ({item['date']}): {item['score']}")

        print(f"\nInvalid scores: {len(bet_issues['invalid_scores'])}")
        for item in bet_issues['invalid_scores'][:5]:
            print(f"  - {item['teams']} ({item['date']}): {item['score']}")

        print(f"\nInconsistent games: {len(bet_issues['inconsistent_games'])}")
        for item in bet_issues['inconsistent_games'][:5]:
            print(f"  - {item['game']}: {item['scores']}")

        print("\n" + "=" * 60)
        print("TRAINING DATA ANALYSIS")
        print("=" * 60)

        training_issues = cleaner.analyze_training_data()
        print(f"Total records: {training_issues['total_records']}")
        print(f"Missing scores: {len(training_issues['missing_scores'])}")
        print(f"Invalid scores: {len(training_issues['invalid_scores'])}")
        print(f"Missing features: {len(training_issues['missing_features'])}")
        print(f"Large errors (>30 pts): {len(training_issues['large_errors'])}")


def run_data_cleaning(dry_run: bool = True):
    """Run data cleaning operations."""
    with DataCleaner() as cleaner:
        print(f"\n{'DRY RUN - ' if dry_run else ''}DATA CLEANING")
        print("=" * 60)

        results = cleaner.clean_all(dry_run)

        print(f"\nFuture-graded bets to reset: {len(results['future_graded_reset'])}")
        print(f"Invalid-score bets to reset: {len(results['invalid_scores_reset'])}")
        print(f"Inconsistent-game bets to reset: {len(results['inconsistent_games_reset'])}")
        print(f"Invalid training records to delete: {results['invalid_training_deleted']}")

        print(f"\nTOTAL BETS TO RESET: {results['summary']['total_bets_reset']}")
        print(f"TOTAL TRAINING RECORDS TO DELETE: {results['summary']['training_records_deleted']}")

        if dry_run:
            print("\n*** This was a DRY RUN - no changes were made ***")
            print("Run with dry_run=False to apply changes")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        dry_run = "--dry-run" in sys.argv or "-n" in sys.argv
        run_data_cleaning(dry_run)
    else:
        run_data_analysis()
