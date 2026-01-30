# ML module - gracefully handles missing dependencies for Vercel deployment
try:
    from .predictor import MLPredictor, PredictionResult
    from .xgboost_model import XGBoostPredictor, FEATURE_COLUMNS
    ML_AVAILABLE = True
except ImportError:
    # ML packages not installed (e.g., on Vercel)
    MLPredictor = None
    PredictionResult = None
    XGBoostPredictor = None
    FEATURE_COLUMNS = []
    ML_AVAILABLE = False

__all__ = [
    "MLPredictor",
    "PredictionResult",
    "XGBoostPredictor",
    "FEATURE_COLUMNS",
    "ML_AVAILABLE",
]
