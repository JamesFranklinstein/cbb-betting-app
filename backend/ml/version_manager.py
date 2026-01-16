"""
Model Version Manager

Manages ML model versions, A/B testing, and rollback capabilities.
Tracks performance metrics for each model version.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """
    Manages model versions, A/B testing, and rollback.

    Features:
    - Create and track model versions
    - Activate/deactivate versions
    - Track live performance metrics
    - Compare version performance
    - Rollback to previous versions
    """

    def __init__(self, model_dir: str, session: Session):
        """
        Initialize version manager.

        Args:
            model_dir: Directory where models are stored
            session: SQLAlchemy database session
        """
        self.model_dir = model_dir
        self.session = session

    def create_version(
        self,
        model_type: str,
        features: List[str],
        hyperparameters: Dict[str, Any],
        training_metrics: Dict[str, float],
        model_path: str
    ):
        """
        Create a new model version record.

        Args:
            model_type: Type of model (e.g., 'PyTorch', 'sklearn')
            features: List of feature names used
            hyperparameters: Model hyperparameters
            training_metrics: Metrics from training (accuracy, brier_score, etc.)
            model_path: Path to saved model file

        Returns:
            MLModelVersion model instance
        """
        from models.database import MLModelVersion

        # Generate version string
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        version = f"v{timestamp}_{model_type.lower()}"

        model_version = MLModelVersion(
            version=version,
            model_type=model_type,
            features_used=features,
            hyperparameters=hyperparameters,
            train_accuracy=training_metrics.get('accuracy'),
            val_accuracy=training_metrics.get('accuracy'),
            train_log_loss=training_metrics.get('log_loss'),
            val_log_loss=training_metrics.get('log_loss'),
            brier_score=training_metrics.get('brier_score'),
            spread_mae=training_metrics.get('spread_mae'),
            total_mae=training_metrics.get('total_mae'),
            is_active=False,
            trained_at=datetime.utcnow(),
            model_path=model_path,
        )

        self.session.add(model_version)
        self.session.commit()

        logger.info(f"Created model version: {version}")
        return model_version

    def activate_version(self, version: str) -> bool:
        """
        Activate a specific model version (deactivates all others).

        Args:
            version: Version string to activate

        Returns:
            True if successful, False if version not found
        """
        from models.database import MLModelVersion

        # Deactivate all versions
        self.session.query(MLModelVersion).update(
            {MLModelVersion.is_active: False}
        )

        # Activate specified version
        model = self.session.query(MLModelVersion).filter(
            MLModelVersion.version == version
        ).first()

        if model:
            model.is_active = True
            self.session.commit()
            logger.info(f"Activated model version: {version}")
            return True
        else:
            self.session.rollback()
            logger.warning(f"Model version not found: {version}")
            return False

    def deactivate_all(self) -> None:
        """Deactivate all model versions."""
        from models.database import MLModelVersion

        self.session.query(MLModelVersion).update(
            {MLModelVersion.is_active: False}
        )
        self.session.commit()
        logger.info("Deactivated all model versions")

    def get_active_version(self):
        """
        Get the currently active model version.

        Returns:
            MLModelVersion model or None
        """
        from models.database import MLModelVersion

        return self.session.query(MLModelVersion).filter(
            MLModelVersion.is_active == True
        ).first()

    def get_version(self, version: str):
        """
        Get a specific model version by version string.

        Args:
            version: Version string

        Returns:
            MLModelVersion model or None
        """
        from models.database import MLModelVersion

        return self.session.query(MLModelVersion).filter(
            MLModelVersion.version == version
        ).first()

    def get_latest_version(self, model_type: str = None):
        """
        Get the most recent model version.

        Args:
            model_type: Optional filter by model type

        Returns:
            MLModelVersion model or None
        """
        from models.database import MLModelVersion

        query = self.session.query(MLModelVersion).order_by(
            MLModelVersion.trained_at.desc()
        )

        if model_type:
            query = query.filter(MLModelVersion.model_type == model_type)

        return query.first()

    def list_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List all model versions with summary info.

        Args:
            limit: Maximum versions to return

        Returns:
            List of version summary dicts
        """
        from models.database import MLModelVersion

        versions = self.session.query(MLModelVersion).order_by(
            MLModelVersion.created_at.desc()
        ).limit(limit).all()

        return [
            {
                'version': v.version,
                'model_type': v.model_type,
                'brier_score': v.brier_score,
                'accuracy': v.val_accuracy,
                'spread_mae': v.spread_mae,
                'total_mae': v.total_mae,
                'live_accuracy': v.live_accuracy,
                'live_roi': v.live_roi,
                'total_predictions': v.total_predictions,
                'is_active': v.is_active,
                'trained_at': v.trained_at.isoformat() if v.trained_at else None,
                'model_path': v.model_path,
            }
            for v in versions
        ]

    def update_live_performance(
        self,
        version: str,
        accuracy: float = None,
        roi: float = None,
        predictions_count: int = None
    ) -> bool:
        """
        Update live performance metrics for a version.

        Args:
            version: Version string
            accuracy: Live prediction accuracy
            roi: Live return on investment
            predictions_count: Number of predictions made

        Returns:
            True if successful, False if version not found
        """
        from models.database import MLModelVersion

        model = self.session.query(MLModelVersion).filter(
            MLModelVersion.version == version
        ).first()

        if not model:
            logger.warning(f"Model version not found: {version}")
            return False

        if accuracy is not None:
            model.live_accuracy = accuracy
        if roi is not None:
            model.live_roi = roi
        if predictions_count is not None:
            model.total_predictions = predictions_count

        self.session.commit()
        logger.info(f"Updated live performance for {version}")
        return True

    def increment_predictions(self, version: str, count: int = 1) -> bool:
        """
        Increment the prediction count for a version.

        Args:
            version: Version string
            count: Number to increment by

        Returns:
            True if successful
        """
        from models.database import MLModelVersion

        model = self.session.query(MLModelVersion).filter(
            MLModelVersion.version == version
        ).first()

        if not model:
            return False

        model.total_predictions = (model.total_predictions or 0) + count
        self.session.commit()
        return True

    def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            version_a: First version to compare
            version_b: Second version to compare

        Returns:
            Comparison dict with metrics for both versions
        """
        from models.database import MLModelVersion

        model_a = self.session.query(MLModelVersion).filter(
            MLModelVersion.version == version_a
        ).first()
        model_b = self.session.query(MLModelVersion).filter(
            MLModelVersion.version == version_b
        ).first()

        if not model_a or not model_b:
            return {'error': 'One or both versions not found'}

        def get_metrics(model):
            return {
                'version': model.version,
                'model_type': model.model_type,
                'training': {
                    'brier_score': model.brier_score,
                    'accuracy': model.val_accuracy,
                    'log_loss': model.val_log_loss,
                    'spread_mae': model.spread_mae,
                    'total_mae': model.total_mae,
                },
                'live': {
                    'accuracy': model.live_accuracy,
                    'roi': model.live_roi,
                    'predictions': model.total_predictions,
                }
            }

        metrics_a = get_metrics(model_a)
        metrics_b = get_metrics(model_b)

        # Calculate deltas
        def calc_delta(a_val, b_val, lower_is_better=False):
            if a_val is None or b_val is None:
                return None
            delta = b_val - a_val
            return -delta if lower_is_better else delta

        comparison = {
            'version_a': metrics_a,
            'version_b': metrics_b,
            'deltas': {
                'brier_score': calc_delta(
                    model_a.brier_score, model_b.brier_score, lower_is_better=True
                ),
                'accuracy': calc_delta(model_a.val_accuracy, model_b.val_accuracy),
                'spread_mae': calc_delta(
                    model_a.spread_mae, model_b.spread_mae, lower_is_better=True
                ),
                'live_accuracy': calc_delta(model_a.live_accuracy, model_b.live_accuracy),
                'live_roi': calc_delta(model_a.live_roi, model_b.live_roi),
            },
            'recommendation': self._recommend_version(model_a, model_b),
        }

        return comparison

    def _recommend_version(self, model_a, model_b) -> str:
        """
        Recommend which version to use based on metrics.

        Prioritizes:
        1. Live performance (if available)
        2. Validation Brier score (calibration is critical for betting)
        """
        # If we have live data for both, use that
        if model_a.live_accuracy and model_b.live_accuracy:
            if model_a.live_accuracy > model_b.live_accuracy + 0.02:
                return f"Recommend {model_a.version} (better live accuracy)"
            elif model_b.live_accuracy > model_a.live_accuracy + 0.02:
                return f"Recommend {model_b.version} (better live accuracy)"

        # Fall back to Brier score
        if model_a.brier_score and model_b.brier_score:
            if model_a.brier_score < model_b.brier_score - 0.005:
                return f"Recommend {model_a.version} (better calibration)"
            elif model_b.brier_score < model_a.brier_score - 0.005:
                return f"Recommend {model_b.version} (better calibration)"

        return "No clear recommendation - versions perform similarly"

    def get_best_version(self, metric: str = 'brier_score'):
        """
        Get the best performing version by a specific metric.

        Args:
            metric: Metric to optimize ('brier_score', 'accuracy', 'live_roi')

        Returns:
            MLModelVersion model or None
        """
        from models.database import MLModelVersion

        query = self.session.query(MLModelVersion)

        if metric == 'brier_score':
            # Lower is better
            query = query.filter(MLModelVersion.brier_score.isnot(None))
            query = query.order_by(MLModelVersion.brier_score.asc())
        elif metric == 'accuracy':
            query = query.filter(MLModelVersion.val_accuracy.isnot(None))
            query = query.order_by(MLModelVersion.val_accuracy.desc())
        elif metric == 'live_roi':
            query = query.filter(MLModelVersion.live_roi.isnot(None))
            query = query.order_by(MLModelVersion.live_roi.desc())
        elif metric == 'live_accuracy':
            query = query.filter(MLModelVersion.live_accuracy.isnot(None))
            query = query.order_by(MLModelVersion.live_accuracy.desc())
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return query.first()

    def cleanup_old_versions(
        self,
        keep_count: int = 5,
        keep_active: bool = True
    ) -> int:
        """
        Remove old model versions and their files.

        Args:
            keep_count: Number of versions to keep
            keep_active: Whether to keep the active version

        Returns:
            Number of versions deleted
        """
        from models.database import MLModelVersion

        # Get versions to keep (most recent)
        keep_query = self.session.query(MLModelVersion).order_by(
            MLModelVersion.created_at.desc()
        ).limit(keep_count)
        keep_ids = [v.id for v in keep_query.all()]

        # Find versions to delete
        delete_query = self.session.query(MLModelVersion).filter(
            ~MLModelVersion.id.in_(keep_ids)
        )

        if keep_active:
            delete_query = delete_query.filter(MLModelVersion.is_active == False)

        to_delete = delete_query.all()
        deleted_count = 0

        for version in to_delete:
            # Delete model file if exists
            if version.model_path and os.path.exists(version.model_path):
                try:
                    os.remove(version.model_path)
                    logger.info(f"Deleted model file: {version.model_path}")
                except OSError as e:
                    logger.warning(f"Could not delete model file: {e}")

            # Delete database record
            self.session.delete(version)
            deleted_count += 1

        self.session.commit()
        logger.info(f"Cleaned up {deleted_count} old model versions")
        return deleted_count

    def export_version_history(self) -> List[Dict[str, Any]]:
        """
        Export complete version history for analysis.

        Returns:
            List of all version records with full details
        """
        from models.database import MLModelVersion

        versions = self.session.query(MLModelVersion).order_by(
            MLModelVersion.created_at.desc()
        ).all()

        return [
            {
                'version': v.version,
                'model_type': v.model_type,
                'features_used': v.features_used,
                'hyperparameters': v.hyperparameters,
                'train_accuracy': v.train_accuracy,
                'val_accuracy': v.val_accuracy,
                'brier_score': v.brier_score,
                'calibration_error': v.calibration_error,
                'spread_mae': v.spread_mae,
                'total_mae': v.total_mae,
                'live_accuracy': v.live_accuracy,
                'live_roi': v.live_roi,
                'total_predictions': v.total_predictions,
                'is_active': v.is_active,
                'trained_at': v.trained_at.isoformat() if v.trained_at else None,
                'created_at': v.created_at.isoformat() if v.created_at else None,
                'model_path': v.model_path,
            }
            for v in versions
        ]
