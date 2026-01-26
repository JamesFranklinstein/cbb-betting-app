"""
ML Retraining Scheduler

Automated scheduling for model retraining and feedback collection.
Uses APScheduler for background job execution.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_EXECUTED

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Scheduler for automated ML model retraining.

    Schedules:
    - Daily feedback collection (11:59 PM)
    - Weekly model retraining (Sunday 2:00 AM)

    Uses APScheduler for reliable background job execution.
    """

    def __init__(
        self,
        db_session_factory: Callable,
        model_dir: str = "./models",
        timezone: str = "America/New_York"
    ):
        """
        Initialize the retraining scheduler.

        Args:
            db_session_factory: Callable that returns a new DB session
            model_dir: Directory for model storage
            timezone: Timezone for scheduling (default: US Eastern)
        """
        self.db_session_factory = db_session_factory
        self.model_dir = model_dir
        self.timezone = timezone

        self.scheduler = BackgroundScheduler(timezone=timezone)
        self._setup_job_listeners()

        # Track job history
        self.job_history = []

    def _setup_job_listeners(self):
        """Set up listeners for job events."""
        def job_executed(event):
            self.job_history.append({
                'job_id': event.job_id,
                'status': 'success',
                'time': datetime.now(timezone.utc).isoformat(),
            })
            logger.info(f"Job executed: {event.job_id}")

        def job_error(event):
            self.job_history.append({
                'job_id': event.job_id,
                'status': 'error',
                'error': str(event.exception),
                'time': datetime.now(timezone.utc).isoformat(),
            })
            logger.error(f"Job error: {event.job_id} - {event.exception}")

        self.scheduler.add_listener(job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(job_error, EVENT_JOB_ERROR)

    def start(self):
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("Retraining scheduler started")

    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Retraining scheduler stopped")

    def schedule_feedback_collection(
        self,
        hour: int = 23,
        minute: int = 59
    ):
        """
        Schedule daily feedback collection.

        Args:
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
        """
        self.scheduler.add_job(
            self._run_feedback_collection,
            CronTrigger(hour=hour, minute=minute),
            id='daily_feedback_collection',
            replace_existing=True,
            name='Daily Feedback Collection'
        )
        logger.info(f"Scheduled daily feedback collection at {hour:02d}:{minute:02d}")

    def schedule_weekly_retraining(
        self,
        day_of_week: str = 'sun',
        hour: int = 2,
        minute: int = 0
    ):
        """
        Schedule weekly model retraining.

        Args:
            day_of_week: Day to run ('mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun')
            hour: Hour to run (0-23)
            minute: Minute to run (0-59)
        """
        self.scheduler.add_job(
            self._run_retraining,
            CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
            id='weekly_retraining',
            replace_existing=True,
            name='Weekly Model Retraining'
        )
        logger.info(f"Scheduled weekly retraining on {day_of_week} at {hour:02d}:{minute:02d}")

    def schedule_all(self):
        """Schedule all default jobs."""
        self.schedule_feedback_collection()
        self.schedule_weekly_retraining()

    def _run_feedback_collection(self):
        """Execute feedback collection job."""
        logger.info("Starting scheduled feedback collection")

        session = self.db_session_factory()
        try:
            from .feedback_collector import FeedbackCollector

            collector = FeedbackCollector(session)
            result = collector.run_collection(days_back=1, store=True)

            logger.info(f"Feedback collection complete: {result}")
            return result

        except Exception as e:
            logger.error(f"Feedback collection failed: {e}")
            raise
        finally:
            session.close()

    def _run_retraining(self):
        """Execute model retraining job."""
        logger.info("Starting scheduled model retraining")

        session = self.db_session_factory()
        try:
            from .feedback_collector import FeedbackCollector
            from .trainer import CBBModelTrainer
            from .version_manager import ModelVersionManager
            import pandas as pd

            # Get training data
            collector = FeedbackCollector(session)
            df = collector.get_training_dataframe()

            if len(df) < 100:
                logger.warning(f"Insufficient training data ({len(df)} games). Skipping retraining.")
                return {'status': 'skipped', 'reason': 'insufficient_data', 'games': len(df)}

            # Split data (time-based)
            df = df.sort_values('game_date')
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]

            logger.info(f"Training on {len(train_df)} games, validating on {len(val_df)} games")

            # Train new model
            trainer = CBBModelTrainer(model_dir=self.model_dir)
            metrics = trainer.train(
                train_df=train_df,
                val_df=val_df,
                epochs=100,
                patience=15
            )

            # Calibrate
            trainer.calibrate_temperature(val_df)

            # Save and register version
            version_str = f"v{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_pytorch"
            model_path = trainer.save_model(version_str)

            version_manager = ModelVersionManager(self.model_dir, session)
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

            # Check if new model is better than current active
            active_version = version_manager.get_active_version()
            if active_version:
                if metrics['brier_score'] < (active_version.brier_score or 1.0) - 0.005:
                    version_manager.activate_version(version_str)
                    logger.info(f"Activated new model {version_str} (better Brier score)")
                else:
                    logger.info(f"Kept current model (new Brier score not significantly better)")
            else:
                version_manager.activate_version(version_str)
                logger.info(f"Activated new model {version_str} (no previous active)")

            return {
                'status': 'success',
                'version': version_str,
                'metrics': metrics,
                'training_games': len(train_df),
                'validation_games': len(val_df),
            }

        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            raise
        finally:
            session.close()

    def run_now(self, job_name: str) -> dict:
        """
        Run a job immediately.

        Args:
            job_name: 'feedback' or 'retrain'

        Returns:
            Job result dict
        """
        if job_name == 'feedback':
            return self._run_feedback_collection()
        elif job_name == 'retrain':
            return self._run_retraining()
        else:
            raise ValueError(f"Unknown job: {job_name}")

    def get_status(self) -> dict:
        """
        Get scheduler status and job information.

        Returns:
            Status dict with jobs and history
        """
        jobs = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': next_run.isoformat() if next_run else None,
            })

        return {
            'running': self.scheduler.running,
            'jobs': jobs,
            'recent_history': self.job_history[-10:],
        }


def create_scheduler(
    db_session_factory: Callable,
    model_dir: str = "./models",
    auto_start: bool = True
) -> RetrainingScheduler:
    """
    Factory function to create and optionally start a scheduler.

    Args:
        db_session_factory: Callable that returns a new DB session
        model_dir: Directory for model storage
        auto_start: Whether to start the scheduler immediately

    Returns:
        Configured RetrainingScheduler instance
    """
    scheduler = RetrainingScheduler(
        db_session_factory=db_session_factory,
        model_dir=model_dir
    )

    scheduler.schedule_all()

    if auto_start:
        scheduler.start()

    return scheduler
