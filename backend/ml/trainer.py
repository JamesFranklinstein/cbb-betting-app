"""
Model Training Service for CBB Predictions

Handles training, validation, and calibration of PyTorch models
with support for the feedback loop from historical predictions and results.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

from .pytorch_model import CBBPredictionNet, CalibrationLoss

logger = logging.getLogger(__name__)


class CBBModelTrainer:
    """
    Trainer for CBB prediction models with feedback loop support.

    Handles the complete training pipeline:
    - Feature preparation and normalization
    - Model training with early stopping
    - Temperature calibration for probability accuracy
    - Model checkpoint management
    - Training metrics tracking

    Args:
        model_dir: Directory to save models and checkpoints
        device: PyTorch device ('cuda' or 'cpu')
        height_weight: Weight multiplier for height features (default: 0.33)
    """

    # Feature columns in expected order
    FEATURE_COLUMNS = [
        # Existing features (18)
        "adj_em_diff", "adj_oe_diff", "adj_de_diff", "adj_tempo_diff",
        "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
        "d_efg_pct_diff", "d_to_pct_diff", "d_or_pct_diff", "d_ft_rate_diff",
        "sos_diff", "luck_diff", "home_advantage", "rank_diff",
        "home_win_streak", "away_win_streak",
        # Height features (3)
        "height_diff", "effective_height_diff", "height_vs_tempo",
        # NEW: Situational features (12)
        "days_of_season",           # Time of season (0-1, higher = later)
        "is_conference_game",       # Binary: conference game indicator
        "is_same_conference",       # Binary: teams in same conference
        "home_rest_days",           # Normalized rest days (0-1)
        "away_rest_days",           # Normalized rest days (0-1)
        "rest_advantage",           # Home rest advantage (-1 to 1)
        "home_back_to_back",        # Binary: home on back-to-back
        "away_back_to_back",        # Binary: away on back-to-back
        "travel_distance",          # Normalized travel distance
        "home_conf_strength",       # Home conference strength
        "away_conf_strength",       # Away conference strength
        "conf_strength_diff",       # Conference strength differential
    ]

    def __init__(
        self,
        model_dir: str = "./models",
        device: str = None,
        height_weight: float = 0.33
    ):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.height_weight = height_weight

        # Model and normalization parameters
        self.model: Optional[CBBPredictionNet] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        # Training history
        self.training_history: List[Dict[str, float]] = []

        logger.info(f"CBBModelTrainer initialized on device: {self.device}")

    def prepare_data(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Prepare training data from DataFrame.

        Args:
            df: DataFrame with features and targets
            fit_scaler: If True, fit the scaler to this data

        Returns:
            Tuple of (X tensor, dict of target tensors)
        """
        # Extract available feature columns
        feature_cols = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        missing_cols = [c for c in self.FEATURE_COLUMNS if c not in df.columns]

        if missing_cols:
            logger.warning(f"Missing feature columns: {missing_cols}")
            # Add missing columns with zeros
            for col in missing_cols:
                df[col] = 0.0
            feature_cols = self.FEATURE_COLUMNS

        X = df[feature_cols].fillna(0).values.astype(np.float32)

        # Fit or apply normalization
        if fit_scaler or self.feature_means is None:
            self.feature_means = X.mean(axis=0)
            self.feature_stds = X.std(axis=0)
            # Prevent division by zero
            self.feature_stds[self.feature_stds < 1e-8] = 1.0
            logger.info("Fitted feature scaler")

        X_normalized = (X - self.feature_means) / self.feature_stds

        # Extract targets
        targets = {}
        if 'home_won' in df.columns:
            targets['win'] = torch.tensor(
                df['home_won'].values.astype(np.float32),
                dtype=torch.float32
            )
        if 'actual_spread' in df.columns:
            targets['spread'] = torch.tensor(
                df['actual_spread'].fillna(0).values.astype(np.float32),
                dtype=torch.float32
            )
        if 'actual_total' in df.columns:
            targets['total'] = torch.tensor(
                df['actual_total'].fillna(140).values.astype(np.float32),
                dtype=torch.float32
            )

        X_tensor = torch.tensor(X_normalized, dtype=torch.float32)

        return X_tensor, targets

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        patience: int = 15,
        hidden_sizes: Tuple[int, ...] = (64, 128, 64),
        dropout: float = 0.3,
    ) -> Dict[str, float]:
        """
        Train the PyTorch model with early stopping.

        Args:
            train_df: Training DataFrame with features and targets
            val_df: Validation DataFrame
            epochs: Maximum epochs to train
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            patience: Early stopping patience
            hidden_sizes: Hidden layer sizes for the trunk
            dropout: Dropout rate

        Returns:
            Dictionary of final training metrics
        """
        logger.info(f"Starting training with {len(train_df)} train, {len(val_df)} val samples")

        # Prepare data
        X_train, y_train = self.prepare_data(train_df, fit_scaler=True)
        X_val, y_val = self.prepare_data(val_df, fit_scaler=False)

        # Validate we have targets
        required_targets = ['win', 'spread', 'total']
        for target in required_targets:
            if target not in y_train or target not in y_val:
                raise ValueError(f"Missing required target: {target}")

        # Create data loaders
        train_dataset = TensorDataset(
            X_train,
            y_train['win'],
            y_train['spread'],
            y_train['total']
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True if len(train_dataset) > batch_size else False
        )

        # Initialize model
        self.model = CBBPredictionNet(
            n_features=X_train.shape[1],
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            height_weight=self.height_weight
        ).to(self.device)

        logger.info(f"Model parameters: {self.model.count_parameters():,}")

        # Loss functions
        win_criterion = CalibrationLoss(brier_weight=0.3)
        spread_criterion = nn.SmoothL1Loss()  # Huber loss for robustness
        total_criterion = nn.SmoothL1Loss()

        # Optimizer with weight decay
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
        )

        # Training loop
        best_val_loss = float('inf')
        best_val_brier = float('inf')
        patience_counter = 0
        self.training_history = []

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_losses = []

            for batch in train_loader:
                X_batch, y_win, y_spread, y_total = [
                    b.to(self.device) for b in batch
                ]

                optimizer.zero_grad()
                outputs = self.model(X_batch, return_calibrated=False)

                # Multi-task loss
                loss_win = win_criterion(
                    torch.sigmoid(outputs['win_logits']),
                    y_win
                )
                loss_spread = spread_criterion(outputs['spread'], y_spread)
                loss_total = total_criterion(outputs['total'], y_total)

                # Weight win probability higher for betting applications
                loss = 2.0 * loss_win + loss_spread + loss_total

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())

            # Validation phase
            val_metrics = self._evaluate(X_val, y_val)
            val_loss = val_metrics['combined_loss']

            # Track history
            epoch_metrics = {
                'epoch': epoch,
                'train_loss': np.mean(train_losses),
                'val_loss': val_loss,
                **val_metrics
            }
            self.training_history.append(epoch_metrics)

            # Early stopping check
            if val_metrics['brier_score'] < best_val_brier:
                best_val_brier = val_metrics['brier_score']
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint('best_model.pt')
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

            # Logging
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(
                    f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}, "
                    f"val_brier={val_metrics['brier_score']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )

        # Load best model
        self._load_checkpoint('best_model.pt')

        # Final evaluation
        final_metrics = self._evaluate(X_val, y_val)
        final_metrics['epochs_trained'] = epoch + 1
        final_metrics['best_val_loss'] = best_val_loss

        logger.info(f"Training complete. Final metrics: {final_metrics}")

        return final_metrics

    def _evaluate(
        self,
        X: torch.Tensor,
        y: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate model on validation data.

        Args:
            X: Feature tensor
            y: Dict of target tensors

        Returns:
            Dict of evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            X_device = X.to(self.device)
            outputs = self.model(X_device, return_calibrated=True)

            win_prob = outputs['win_prob'].cpu().numpy()
            spread_pred = outputs['spread'].cpu().numpy()
            total_pred = outputs['total'].cpu().numpy()

            y_win = y['win'].numpy()
            y_spread = y['spread'].numpy()
            y_total = y['total'].numpy()

        # Clip probabilities for log_loss
        win_prob_clipped = np.clip(win_prob, 1e-7, 1 - 1e-7)

        # Calculate metrics
        brier = brier_score_loss(y_win, win_prob)
        logloss = log_loss(y_win, win_prob_clipped)
        accuracy = accuracy_score(y_win, (win_prob > 0.5).astype(int))

        # NEW: Calculate Expected Calibration Error (ECE)
        ece = self._calculate_ece(win_prob, y_win, n_bins=10)

        spread_mae = np.mean(np.abs(spread_pred - y_spread))
        total_mae = np.mean(np.abs(total_pred - y_total))

        # Combined loss for early stopping (weight win prob higher)
        # NEW: Include ECE in combined loss for better calibration
        combined_loss = 2 * brier + 0.5 * ece + spread_mae / 10 + total_mae / 10

        return {
            'brier_score': float(brier),
            'log_loss': float(logloss),
            'accuracy': float(accuracy),
            'ece': float(ece),  # NEW: Expected Calibration Error
            'spread_mae': float(spread_mae),
            'total_mae': float(total_mae),
            'combined_loss': float(combined_loss),
        }

    def _calculate_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures how well-calibrated probability predictions are.
        A perfectly calibrated model would have ECE = 0.

        For betting, good calibration is critical - we need our 60% confidence
        predictions to actually win 60% of the time.

        Formula: ECE = sum_i (n_i / N) * |acc_i - conf_i|
        where acc_i is accuracy in bin i and conf_i is average confidence in bin i

        Args:
            probs: Predicted probabilities
            labels: True binary labels
            n_bins: Number of calibration bins

        Returns:
            ECE value (lower is better, 0 = perfect calibration)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total_samples = len(probs)

        for i in range(n_bins):
            # Find samples in this probability bin
            in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                in_bin = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])

            n_in_bin = in_bin.sum()
            if n_in_bin > 0:
                # Average predicted probability in bin
                avg_confidence = probs[in_bin].mean()
                # Actual accuracy in bin
                avg_accuracy = labels[in_bin].mean()
                # Weighted contribution to ECE
                ece += (n_in_bin / total_samples) * abs(avg_accuracy - avg_confidence)

        return ece

    def calibrate_temperature(
        self,
        val_df: pd.DataFrame,
        max_iter: int = 50
    ) -> float:
        """
        Fine-tune temperature scaling on validation set.

        Uses LBFGS optimization to find optimal temperature that
        minimizes BCE loss on validation probabilities.

        Args:
            val_df: Validation DataFrame
            max_iter: Maximum LBFGS iterations

        Returns:
            Final temperature value
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_val, y_val = self.prepare_data(val_df, fit_scaler=False)
        X_val = X_val.to(self.device)
        y_win = y_val['win'].to(self.device)

        # Freeze all parameters except temperature
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.temperature.temperature.requires_grad = True

        # LBFGS optimizer for temperature
        optimizer = torch.optim.LBFGS(
            [self.model.temperature.temperature],
            lr=0.01,
            max_iter=max_iter
        )

        def closure():
            optimizer.zero_grad()
            outputs = self.model(X_val, return_calibrated=True)
            loss = nn.BCELoss()(outputs['win_prob'], y_win)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Re-enable gradients for all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        final_temp = self.model.temperature.get_temperature()
        logger.info(f"Temperature calibration complete: T={final_temp:.4f}")

        return final_temp

    def predict(
        self,
        features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Make a prediction for a single game.

        Args:
            features: Dict of feature name -> value

        Returns:
            Dict with win_prob, spread, total predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        # Build feature vector in correct order
        feature_vector = np.array([
            features.get(col, 0.0) for col in self.FEATURE_COLUMNS
        ], dtype=np.float32)

        return self.model.predict(
            feature_vector,
            self.feature_means,
            self.feature_stds,
            self.device
        )

    def predict_batch(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Make predictions for multiple games.

        Args:
            df: DataFrame with feature columns

        Returns:
            DataFrame with predictions added
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        X, _ = self.prepare_data(df, fit_scaler=False)

        self.model.eval()
        with torch.no_grad():
            X_device = X.to(self.device)
            outputs = self.model(X_device)

        result = df.copy()
        result['pred_win_prob'] = outputs['win_prob'].cpu().numpy()
        result['pred_spread'] = outputs['spread'].cpu().numpy()
        result['pred_total'] = outputs['total'].cpu().numpy()

        return result

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.model_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'height_weight': self.height_weight,
            'model_config': {
                'n_features': self.model.n_features,
                'hidden_sizes': self.model.hidden_sizes,
                'height_feature_indices': self.model.height_feature_indices,
            }
        }, path)

    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.model_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_means = checkpoint['feature_means']
        self.feature_stds = checkpoint['feature_stds']

    def save_model(self, version: str) -> str:
        """
        Save trained model with version tag.

        Args:
            version: Version string (e.g., 'v20240115_pytorch')

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save.")

        filename = f"cbb_model_{version}.pt"
        path = os.path.join(self.model_dir, filename)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'height_weight': self.height_weight,
            'feature_columns': self.FEATURE_COLUMNS,
            'model_config': {
                'n_features': self.model.n_features,
                'hidden_sizes': self.model.hidden_sizes,
                'height_feature_indices': self.model.height_feature_indices,
            },
            'version': version,
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'training_history': self.training_history,
        }, path)

        logger.info(f"Model saved to {path}")
        return path

    def load_model(self, path: str) -> None:
        """
        Load a trained model from file.

        Args:
            path: Path to saved model file
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Reconstruct model with saved config
        config = checkpoint.get('model_config', {})
        self.model = CBBPredictionNet(
            n_features=config.get('n_features', 21),
            hidden_sizes=config.get('hidden_sizes', (64, 128, 64)),
            height_feature_indices=config.get('height_feature_indices', (18, 19, 20)),
            height_weight=checkpoint.get('height_weight', 0.33),
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_means = checkpoint['feature_means']
        self.feature_stds = checkpoint['feature_stds']
        self.height_weight = checkpoint.get('height_weight', 0.33)

        logger.info(f"Model loaded from {path}")

    def get_calibration_curve(
        self,
        df: pd.DataFrame,
        n_bins: int = 10
    ) -> Dict[str, List[float]]:
        """
        Calculate calibration curve for model evaluation.

        Args:
            df: DataFrame with features and home_won column
            n_bins: Number of probability bins

        Returns:
            Dict with bin_centers, actual_freqs, counts
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        # Get predictions
        X, y = self.prepare_data(df, fit_scaler=False)
        self.model.eval()

        with torch.no_grad():
            X_device = X.to(self.device)
            outputs = self.model(X_device)
            probs = outputs['win_prob'].cpu().numpy()

        actuals = y['win'].numpy()

        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        actual_freqs = []
        counts = []

        for i in range(n_bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if i == n_bins - 1:
                mask = (probs >= bin_edges[i]) & (probs <= bin_edges[i + 1])

            count = mask.sum()
            if count > 0:
                bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                actual_freqs.append(actuals[mask].mean())
                counts.append(int(count))

        return {
            'bin_centers': bin_centers,
            'actual_freqs': actual_freqs,
            'counts': counts,
        }
