"""
Generate AQI Predictions
- Load trained models
- Generate predictions for all locations
- Compare Random Forest vs XGBoost
- Save predictions for Power BI
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_models():
    """Load all trained models."""
    logger.info("Loading trained models...")

    models_dir = Path("models")
    locations = [
        "R_K_Puram_Delhi___DPCC",
        "Punjabi_Bagh_Delhi___DPCC",
        "Anand_Vihar_New_Delhi___DPCC",
        "Zoo_Park_Hyderabad___TSPCB",
    ]

    rf_models = {}
    xgb_models = {}

    for loc in locations:
        rf_models[loc] = joblib.load(models_dir / f"rf_{loc}.pkl")
        xgb_models[loc] = joblib.load(models_dir / f"xgb_{loc}.pkl")
        logger.info(f"  Loaded models for: {loc}")

    return rf_models, xgb_models, locations


def prepare_features(df):
    """Prepare features (same as training)."""
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
    feature_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()

    return feature_cols


def normalize_location_name(name: str) -> str:
    """Normalize location name to match model filename convention."""
    return name.replace(",", "").replace(" ", "_").replace("-", "_")


def main():
    """Main prediction pipeline."""
    logger.info("=" * 80)
    logger.info("Starting AQI Prediction Generation")
    logger.info("=" * 80)

    # Load models
    rf_models, xgb_models, locations = load_models()

    # Load data
    logger.info("\nLoading ML-ready data...")
    df = pd.read_csv("data/processed/ml_ready_data.csv")
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    logger.info(f"Loaded {len(df):,} records")

    # Prepare features
    feature_cols = prepare_features(df)
    logger.info(f"Using {len(feature_cols)} features")

    # Generate predictions for each location
    all_predictions = []

    normalized_location_series = df["location_name"].apply(normalize_location_name)

    for loc in locations:
        logger.info(f"\nGenerating predictions for: {loc}")

        # Filter data for this location
        loc_data = df[normalized_location_series == loc].copy()

        if len(loc_data) == 0:
            logger.warning(f"  No data found for {loc}")
            continue

        # Prepare features
        X = loc_data[feature_cols].ffill().bfill()

        # Predict with both models
        rf_pred = rf_models[loc].predict(X)
        xgb_pred = xgb_models[loc].predict(X)

        # Create predictions dataframe
        pred_df = pd.DataFrame(
            {
                "datetime": loc_data["datetime_utc"].values,
                "location": loc_data["location_name"].values,
                "actual_aqi": loc_data["AQI"].values,
                "predicted_aqi_rf": rf_pred,
                "predicted_aqi_xgb": xgb_pred,
                "error_rf": np.abs(loc_data["AQI"].values - rf_pred),
                "error_xgb": np.abs(loc_data["AQI"].values - xgb_pred),
            }
        )

        all_predictions.append(pred_df)
        logger.info(f"  Generated {len(pred_df):,} predictions")

    if not all_predictions:
        raise RuntimeError("No predictions were generated. Check model files and input data.")

    # Combine all predictions
    predictions = pd.concat(all_predictions, ignore_index=True)

    # Create output directory
    Path("data/predictions").mkdir(exist_ok=True)

    # Save predictions
    output_file = "data/predictions/predictions_all.csv"
    predictions.to_csv(output_file, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total predictions: {len(predictions):,}")
    logger.info("\nRandom Forest Performance:")
    logger.info(f"  Mean Absolute Error: {predictions['error_rf'].mean():.2f}")
    logger.info(f"  Median Absolute Error: {predictions['error_rf'].median():.2f}")
    logger.info("\nXGBoost Performance:")
    logger.info(f"  Mean Absolute Error: {predictions['error_xgb'].mean():.2f}")
    logger.info(f"  Median Absolute Error: {predictions['error_xgb'].median():.2f}")
    logger.info(f"\nSaved to: {output_file}")
    logger.info("=" * 80)
    logger.info("Next step: Export for Power BI (8_export_for_powerbi.py)")


if __name__ == "__main__":
    main()
