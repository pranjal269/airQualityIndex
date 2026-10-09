"""
ML Model Training for AQI Prediction
- Train Random Forest (primary) and XGBoost (secondary) models
- Location-wise prediction (separate model for each location)
- Model evaluation and comparison
"""

import logging
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_ml_data() -> pd.DataFrame:
    """Load ML-ready data."""
    logger.info("Loading ML-ready data...")
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "processed" / "ml_ready_data.csv"
    df = pd.read_csv(data_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    logger.info(f"Loaded {len(df):,} records")
    return df


def prepare_features(df: pd.DataFrame) -> list[str]:
    """Prepare features for modeling."""
    logger.info("Preparing features...")

    exclude_cols = [
        "datetime_utc",
        "location_id",
        "location_name",
        "city",
        "latitude",
        "longitude",
        "AQI",
        "AQI_Category",
        "Dominant_Pollutant",
        "aqi_category",
        "dominant_pollutant",
        "season",
        "time_of_day",
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

    logger.info(f"Selected {len(numeric_cols)} numeric features for modeling")
    return numeric_cols


def train_location_model(
    df_location: pd.DataFrame,
    location_name: str,
    feature_cols: list[str],
    model_type: str = "rf",
):
    """Train model for a specific location."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Training {model_type.upper()} model for: {location_name}")
    logger.info(f"{'=' * 80}")

    X = df_location[feature_cols].copy().ffill().bfill()
    y = df_location["AQI"].copy()

    mask = ~y.isna()
    X = X[mask]
    y = y[mask]

    logger.info(f"Training samples: {len(X):,}")
    logger.info(f"AQI range: {y.min():.1f} - {y.max():.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=False,
    )

    logger.info(f"Train set: {len(X_train):,} samples")
    logger.info(f"Test set: {len(X_test):,} samples")

    if model_type == "rf":
        logger.info("Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    else:
        logger.info("Training XGBoost...")
        model = XGBRegressor(
            n_estimators=100,
            max_depth=10,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)

    logger.info("\nTraining Performance:")
    logger.info(f"  MAE:  {train_mae:.2f}")
    logger.info(f"  RMSE: {train_rmse:.2f}")
    logger.info(f"  R²:   {train_r2:.4f}")

    logger.info("\nTest Performance:")
    logger.info(f"  MAE:  {test_mae:.2f}")
    logger.info(f"  RMSE: {test_rmse:.2f}")
    logger.info(f"  R²:   {test_r2:.4f}")

    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info("\nTop 10 Important Features:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    metrics = {
        "location": location_name,
        "model_type": model_type,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "train_r2": float(train_r2),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "feature_count": len(feature_cols),
    }

    return model, metrics, feature_importance


def save_model(model, location_name: str, model_type: str, script_dir: Path) -> Path:
    """Save trained model."""
    models_dir = script_dir.parent / "models"
    models_dir.mkdir(exist_ok=True)

    clean_name = location_name.replace(",", "").replace(" ", "_").replace("-", "_")
    model_path = models_dir / f"{model_type}_{clean_name}.pkl"

    joblib.dump(model, model_path)
    logger.info(f"Saved model to: {model_path}")

    return model_path


def main() -> None:
    """Main training pipeline."""
    logger.info("=" * 80)
    logger.info("Starting ML Model Training")
    logger.info("=" * 80)

    df = load_ml_data()
    feature_cols = prepare_features(df)

    locations = df["location_name"].unique()
    logger.info(f"\nTraining models for {len(locations)} locations:")
    for loc in locations:
        logger.info(f"  - {loc}")

    all_metrics = []
    all_feature_importance = {}
    script_dir = Path(__file__).parent

    for location in locations:
        df_location = df[df["location_name"] == location].copy()

        rf_model, rf_metrics, rf_importance = train_location_model(
            df_location,
            location,
            feature_cols,
            model_type="rf",
        )
        save_model(rf_model, location, "rf", script_dir)
        all_metrics.append(rf_metrics)
        if rf_importance is not None:
            all_feature_importance[f"rf_{location}"] = rf_importance

        xgb_model, xgb_metrics, xgb_importance = train_location_model(
            df_location,
            location,
            feature_cols,
            model_type="xgb",
        )
        save_model(xgb_model, location, "xgb", script_dir)
        all_metrics.append(xgb_metrics)
        if xgb_importance is not None:
            all_feature_importance[f"xgb_{location}"] = xgb_importance

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = script_dir.parent / "models" / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nSaved metrics to: {metrics_path}")

    importance_dir = script_dir.parent / "models" / "feature_importance"
    importance_dir.mkdir(exist_ok=True)
    for key, importance_df in all_feature_importance.items():
        importance_path = importance_dir / f"{key}.csv"
        importance_df.to_csv(importance_path, index=False)
    logger.info(f"Saved feature importance to: {importance_dir}")

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    rf_metrics_df = metrics_df[metrics_df["model_type"] == "rf"]
    logger.info("\nRandom Forest Performance:")
    logger.info(f"  Average Test MAE:  {rf_metrics_df['test_mae'].mean():.2f}")
    logger.info(f"  Average Test RMSE: {rf_metrics_df['test_rmse'].mean():.2f}")
    logger.info(f"  Average Test R²:   {rf_metrics_df['test_r2'].mean():.4f}")

    xgb_metrics_df = metrics_df[metrics_df["model_type"] == "xgb"]
    logger.info("\nXGBoost Performance:")
    logger.info(f"  Average Test MAE:  {xgb_metrics_df['test_mae'].mean():.2f}")
    logger.info(f"  Average Test RMSE: {xgb_metrics_df['test_rmse'].mean():.2f}")
    logger.info(f"  Average Test R²:   {xgb_metrics_df['test_r2'].mean():.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Model Training Complete!")
    logger.info("=" * 80)
    logger.info("Next step: Generate predictions (7_generate_predictions.py)")


if __name__ == "__main__":
    main()
