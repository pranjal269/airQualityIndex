"""
Export Data for Power BI
- Format predictions for Power BI
- Add AQI categories and colors
- Create location metadata
- Export 4 CSV files
"""

from pathlib import Path
import logging

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_aqi_category(aqi):
    """Get AQI category based on Indian CPCB standard."""
    if pd.isna(aqi):
        return "Unknown"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 200:
        return "Poor"
    if aqi <= 300:
        return "Very Poor"
    return "Severe"


def main():
    """Main export pipeline."""
    logger.info("=" * 80)
    logger.info("Starting Power BI Export")
    logger.info("=" * 80)

    logger.info("\nLoading predictions...")
    predictions = pd.read_csv("data/predictions/predictions_all.csv")
    logger.info("Loaded %s predictions", f"{len(predictions):,}")

    if "datetime" in predictions.columns:
        predictions["datetime"] = pd.to_datetime(predictions["datetime"], errors="coerce")

    logger.info("Adding AQI categories...")
    predictions["aqi_category"] = predictions["actual_aqi"].apply(get_aqi_category)
    predictions["predicted_category_rf"] = predictions["predicted_aqi_rf"].apply(get_aqi_category)
    predictions["predicted_category_xgb"] = predictions["predicted_aqi_xgb"].apply(get_aqi_category)

    category_colors = {
        "Good": "#00E400",
        "Moderate": "#FFFF00",
        "Poor": "#FF7E00",
        "Very Poor": "#FF0000",
        "Severe": "#8F3F97",
        "Unknown": "#CCCCCC",
    }
    predictions["color"] = predictions["aqi_category"].map(category_colors)

    logger.info("Creating location metadata...")
    locations_df = (
        predictions.groupby("location", as_index=False)
        .agg(
            total_records=("actual_aqi", "count"),
            avg_aqi=("actual_aqi", "mean"),
            min_aqi=("actual_aqi", "min"),
            max_aqi=("actual_aqi", "max"),
        )
        .rename(columns={"location": "location_name"})
    )

    ml_data = pd.read_csv("data/processed/ml_ready_data.csv")
    coord_cols = [c for c in ["location_name", "latitude", "longitude", "city"] if c in ml_data.columns]
    if "location_name" in coord_cols and len(coord_cols) > 1:
        location_coords = ml_data[coord_cols].drop_duplicates(subset=["location_name"])
        locations_df = locations_df.merge(location_coords, on="location_name", how="left")
    else:
        logger.warning("Location coordinates/city not available in ml_ready_data.csv")

    logger.info("Loading model metrics...")
    metrics = pd.read_csv("models/model_metrics.csv")

    output_dir = Path("data/powerbi")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nExporting files...")
    historical_path = output_dir / "historical_aqi.csv"
    predictions_path = output_dir / "predictions.csv"
    locations_path = output_dir / "locations.csv"
    metrics_path = output_dir / "model_metrics.csv"

    predictions.to_csv(historical_path, index=False)
    logger.info("  OK historical_aqi.csv")

    predictions.to_csv(predictions_path, index=False)
    logger.info("  OK predictions.csv")

    locations_df.to_csv(locations_path, index=False)
    logger.info("  OK locations.csv")

    metrics.to_csv(metrics_path, index=False)
    logger.info("  OK model_metrics.csv")

    logger.info("\n" + "=" * 80)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 80)
    logger.info("Exported 4 files to: %s", output_dir)
    logger.info("  - historical_aqi.csv (%s rows)", f"{len(predictions):,}")
    logger.info("  - predictions.csv (%s rows)", f"{len(predictions):,}")
    logger.info("  - locations.csv (%s rows)", len(locations_df))
    logger.info("  - model_metrics.csv (%s rows)", len(metrics))
    logger.info("=" * 80)
    logger.info("Next step: Create Power BI Dashboard")


if __name__ == "__main__":
    main()
