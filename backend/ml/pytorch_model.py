"""
PyTorch Neural Network Model for CBB Predictions

Multi-head architecture with shared feature extraction and task-specific heads
for win probability, spread prediction, and total prediction.

Features temperature scaling for well-calibrated probabilities critical for betting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np


class TemperatureScaling(nn.Module):
    """
    Temperature scaling layer for probability calibration.

    Post-hoc calibration technique that learns a single temperature parameter
    to scale logits before sigmoid, improving probability calibration without
    affecting the ranking of predictions.

    Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017)
    """

    def __init__(self, initial_temperature: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


class CalibrationLoss(nn.Module):
    """
    Combined loss function for probability calibration.

    Combines binary cross-entropy with Brier score for better calibration.
    The Brier score component penalizes overconfident predictions.

    Args:
        brier_weight: Weight for Brier score component (default: 0.3)
    """

    def __init__(self, brier_weight: float = 0.3):
        super().__init__()
        self.brier_weight = brier_weight
        self.bce = nn.BCELoss()

    def forward(
        self,
        prob: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined BCE + Brier score loss.

        Args:
            prob: Predicted probabilities [batch_size]
            target: True binary labels [batch_size]

        Returns:
            Combined loss value
        """
        bce_loss = self.bce(prob, target.float())
        brier_loss = F.mse_loss(prob, target.float())
        return bce_loss + self.brier_weight * brier_loss


class CBBPredictionNet(nn.Module):
    """
    Multi-head neural network for college basketball predictions.

    Architecture:
    - Shared trunk for feature extraction (learns common representations)
    - Separate heads for: win probability, spread, total
    - Temperature scaling for calibrated probabilities
    - Height feature weighting to give height features less influence

    Args:
        n_features: Number of input features (default: 21)
        hidden_sizes: Tuple of hidden layer sizes for trunk (default: (64, 128, 64))
        dropout: Dropout rate (default: 0.3)
        height_feature_indices: Indices of height features to apply reduced weighting
        height_weight: Weight multiplier for height features (default: 0.33 for 1/3 impact)
    """

    # Default feature order (must match trainer)
    DEFAULT_FEATURE_COLUMNS = [
        # Existing features (18)
        "adj_em_diff", "adj_oe_diff", "adj_de_diff", "adj_tempo_diff",
        "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
        "d_efg_pct_diff", "d_to_pct_diff", "d_or_pct_diff", "d_ft_rate_diff",
        "sos_diff", "luck_diff", "home_advantage", "rank_diff",
        "home_win_streak", "away_win_streak",
        # Height features (3) - NEW
        "height_diff", "effective_height_diff", "height_vs_tempo",
    ]

    def __init__(
        self,
        n_features: int = 21,
        hidden_sizes: Tuple[int, ...] = (64, 128, 64),
        dropout: float = 0.3,
        height_feature_indices: Tuple[int, ...] = (18, 19, 20),
        height_weight: float = 0.33
    ):
        super().__init__()

        self.n_features = n_features
        self.height_feature_indices = height_feature_indices
        self.height_weight = height_weight
        self.hidden_sizes = hidden_sizes

        # Input normalization
        self.input_bn = nn.BatchNorm1d(n_features)
        self.input_dropout = nn.Dropout(0.1)

        # Build shared trunk
        trunk_layers = []
        in_size = n_features
        for hidden_size in hidden_sizes:
            trunk_layers.extend([
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
            ])
            in_size = hidden_size
        trunk_layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*trunk_layers)

        # Task-specific heads
        head_hidden = 32
        trunk_out = hidden_sizes[-1]

        # Win probability head (binary classification)
        self.win_prob_head = nn.Sequential(
            nn.Linear(trunk_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )
        self.temperature = TemperatureScaling()

        # Spread prediction head (regression)
        self.spread_head = nn.Sequential(
            nn.Linear(trunk_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Total prediction head (regression)
        self.total_head = nn.Sequential(
            nn.Linear(trunk_out, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform for better convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def apply_height_weighting(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply reduced weighting to height features.

        This implements the user's requirement for height to have "very little weight"
        by scaling height features to 1/3 of their normalized values.

        Args:
            x: Input tensor [batch_size, n_features]

        Returns:
            Tensor with height features scaled down
        """
        x = x.clone()
        for idx in self.height_feature_indices:
            if idx < x.shape[1]:
                x[:, idx] *= self.height_weight
        return x

    def forward(
        self,
        x: torch.Tensor,
        return_calibrated: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input features [batch_size, n_features]
            return_calibrated: If True, apply temperature scaling to win probability

        Returns:
            Dict with keys:
                - 'win_prob': Calibrated win probability [batch_size]
                - 'spread': Predicted spread [batch_size]
                - 'total': Predicted total [batch_size]
                - 'win_logits': Raw logits before calibration [batch_size]
        """
        # Apply height weighting
        x = self.apply_height_weighting(x)

        # Input processing
        x = self.input_bn(x)
        x = self.input_dropout(x)

        # Shared feature extraction
        shared = self.trunk(x)

        # Task-specific predictions
        win_logits = self.win_prob_head(shared)
        spread = self.spread_head(shared)
        total = self.total_head(shared)

        # Calibrate win probability
        if return_calibrated:
            calibrated_logits = self.temperature(win_logits)
            win_prob = torch.sigmoid(calibrated_logits)
        else:
            win_prob = torch.sigmoid(win_logits)

        return {
            'win_prob': win_prob.squeeze(-1),
            'spread': spread.squeeze(-1),
            'total': total.squeeze(-1),
            'win_logits': win_logits.squeeze(-1),
        }

    def predict(
        self,
        features: np.ndarray,
        feature_means: np.ndarray,
        feature_stds: np.ndarray,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Make a single prediction from raw features.

        Convenience method that handles normalization and conversion.

        Args:
            features: Raw feature values [n_features]
            feature_means: Mean values for normalization
            feature_stds: Std values for normalization
            device: Device to run prediction on

        Returns:
            Dict with win_prob, spread, total predictions
        """
        self.eval()

        # Normalize features
        features_normalized = (features - feature_means) / (feature_stds + 1e-8)

        # Convert to tensor
        x = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
        x = x.to(device)

        # Predict
        with torch.no_grad():
            outputs = self(x)

        return {
            'win_prob': outputs['win_prob'].item(),
            'spread': outputs['spread'].item(),
            'total': outputs['total'].item(),
        }

    def get_feature_importance(
        self,
        feature_names: List[str] = None
    ) -> Dict[str, float]:
        """
        Estimate feature importance from first layer weights.

        This is a rough approximation based on L1 norm of input layer weights.
        For more accurate importance, use SHAP or permutation importance.

        Args:
            feature_names: List of feature names (uses DEFAULT_FEATURE_COLUMNS if None)

        Returns:
            Dict mapping feature names to importance scores
        """
        if feature_names is None:
            feature_names = self.DEFAULT_FEATURE_COLUMNS[:self.n_features]

        # Get first linear layer weights
        first_layer = self.trunk[0]
        if isinstance(first_layer, nn.Linear):
            weights = first_layer.weight.data.cpu().numpy()
            importance = np.abs(weights).sum(axis=0)
            importance = importance / importance.sum()

            return {
                name: float(imp)
                for name, imp in zip(feature_names, importance)
            }

        return {}

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple CBBPredictionNet models.

    Combines predictions from multiple models for more robust predictions.
    Can be used for uncertainty estimation via prediction variance.
    """

    def __init__(
        self,
        models: List[CBBPredictionNet],
        weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        if weights is None:
            weights = [1.0 / self.n_models] * self.n_models
        self.weights = weights

    def forward(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass averaging predictions from all models.

        Returns:
            Dict with averaged predictions and uncertainty estimates
        """
        all_probs = []
        all_spreads = []
        all_totals = []

        for model, weight in zip(self.models, self.weights):
            outputs = model(x)
            all_probs.append(outputs['win_prob'] * weight)
            all_spreads.append(outputs['spread'] * weight)
            all_totals.append(outputs['total'] * weight)

        # Stack and compute mean/std
        probs = torch.stack(all_probs, dim=0)
        spreads = torch.stack(all_spreads, dim=0)
        totals = torch.stack(all_totals, dim=0)

        return {
            'win_prob': probs.sum(dim=0),
            'spread': spreads.sum(dim=0),
            'total': totals.sum(dim=0),
            'win_prob_std': probs.std(dim=0) if self.n_models > 1 else torch.zeros_like(probs[0]),
            'spread_std': spreads.std(dim=0) if self.n_models > 1 else torch.zeros_like(spreads[0]),
            'total_std': totals.std(dim=0) if self.n_models > 1 else torch.zeros_like(totals[0]),
        }
