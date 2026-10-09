"""
Feature Engineering & AQI Calculation
- Pivot data to wide format (parameters as columns)
- Calculate AQI using Indian CPCB formula
- Create lag features (1h, 3h, 6h, 12h, 24h)
- Create rolling features (3h, 6h, 12h, 24h)
- Prepare ML-ready dataset
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cleaned_data():
    """Load cleaned data from preprocessing"""
    logger.info("Loading cleaned data...")
    # Use Path to handle relative paths correctly
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'data' / 'processed' / 'cleaned_data_final.csv'
    df = pd.read_csv(data_path)
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    logger.info(f"Loaded {len(df):,} records")
    return df


def pivot_to_wide_format(df):
    """Pivot from long to wide format (parameters as columns)"""
    logger.info("Pivoting to wide format...")
    
    # Pivot: one row per datetime-location with all parameters as columns
    df_wide = df.pivot_table(
        index=['datetime_utc', 'location_id', 'location_name', 'city', 
               'latitude', 'longitude', 'hour', 'day_of_week', 'month', 
               'season', 'is_weekend'],
        columns='parameter',
        values='value',
        aggfunc='first'  # Take first value if duplicates
    ).reset_index()
    
    # Flatten column names
    df_wide.columns.name = None
    
    logger.info(f"Wide format: {len(df_wide):,} rows, {len(df_wide.columns)} columns")
    logger.info(f"Date range: {df_wide['datetime_utc'].min()} to {df_wide['datetime_utc'].max()}")
    
    return df_wide


def calculate_aqi_subindex(concentration, pollutant):
    """
    Calculate AQI sub-index for a pollutant using Indian CPCB breakpoints
    
    AQI Categories:
    0-50: Good
    51-100: Moderate
    101-200: Poor
    201-300: Very Poor
    301-500: Severe
    """
    
    # Indian CPCB AQI Breakpoints
    breakpoints = {
        'pm25': [
            (0, 30, 0, 50),      # Good
            (31, 60, 51, 100),   # Moderate
            (61, 90, 101, 200),  # Poor
            (91, 120, 201, 300), # Very Poor
            (121, 250, 301, 400),# Severe
            (251, 500, 401, 500) # Severe+
        ],
        'pm10': [
            (0, 50, 0, 50),
            (51, 100, 51, 100),
            (101, 250, 101, 200),
            (251, 350, 201, 300),
            (351, 430, 301, 400),
            (431, 600, 401, 500)
        ],
        'no2': [
            (0, 40, 0, 50),
            (41, 80, 51, 100),
            (81, 180, 101, 200),
            (181, 280, 201, 300),
            (281, 400, 301, 400),
            (401, 500, 401, 500)
        ],
        'so2': [
            (0, 40, 0, 50),
            (41, 80, 51, 100),
            (81, 380, 101, 200),
            (381, 800, 201, 300),
            (801, 1600, 301, 400),
            (1601, 2000, 401, 500)
        ],
        'co': [
            (0, 1.0, 0, 50),
            (1.1, 2.0, 51, 100),
            (2.1, 10, 101, 200),
            (11, 17, 201, 300),
            (18, 34, 301, 400),
            (35, 50, 401, 500)
        ],
        'o3': [
            (0, 50, 0, 50),
            (51, 100, 51, 100),
            (101, 168, 101, 200),
            (169, 208, 201, 300),
            (209, 748, 301, 400),
            (749, 1000, 401, 500)
        ]
    }
    
    if pd.isna(concentration) or pollutant not in breakpoints:
        return np.nan
    
    # Find the appropriate breakpoint
    for bp_low, bp_high, aqi_low, aqi_high in breakpoints[pollutant]:
        if bp_low <= concentration <= bp_high:
            # Linear interpolation
            aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
            return round(aqi)
    
    # If concentration exceeds all breakpoints, return max AQI
    return 500


def calculate_aqi(df_wide):
    """Calculate overall AQI (maximum of all sub-indices)"""
    logger.info("Calculating AQI values...")
    
    # Calculate sub-indices for each pollutant
    pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
    
    for pollutant in pollutants:
        if pollutant in df_wide.columns:
            df_wide[f'aqi_{pollutant}'] = df_wide[pollutant].apply(
                lambda x: calculate_aqi_subindex(x, pollutant)
            )
    
    # Overall AQI is the maximum of all sub-indices
    aqi_columns = [f'aqi_{p}' for p in pollutants if f'aqi_{p}' in df_wide.columns]
    df_wide['AQI'] = df_wide[aqi_columns].max(axis=1)
    
    # Determine AQI category
    def get_aqi_category(aqi):
        if pd.isna(aqi):
            return 'Unknown'
        elif aqi <= 50:
            return 'Good'
        elif aqi <= 100:
            return 'Moderate'
        elif aqi <= 200:
            return 'Poor'
        elif aqi <= 300:
            return 'Very Poor'
        else:
            return 'Severe'
    
    df_wide['AQI_Category'] = df_wide['AQI'].apply(get_aqi_category)
    
    # Identify dominant pollutant (which one determines the AQI)
    def get_dominant_pollutant(row):
        aqi_values = {p: row[f'aqi_{p}'] for p in pollutants if f'aqi_{p}' in row.index and not pd.isna(row[f'aqi_{p}'])}
        if not aqi_values:
            return 'Unknown'
        return max(aqi_values, key=aqi_values.get).upper()
    
    df_wide['Dominant_Pollutant'] = df_wide.apply(get_dominant_pollutant, axis=1)
    
    logger.info(f"AQI calculated for {len(df_wide):,} records")
    logger.info(f"AQI range: {df_wide['AQI'].min():.0f} - {df_wide['AQI'].max():.0f}")
    logger.info(f"\nAQI Category distribution:")
    for cat, count in df_wide['AQI_Category'].value_counts().items():
        logger.info(f"  {cat}: {count:,} ({count/len(df_wide)*100:.1f}%)")
    
    return df_wide


def create_lag_features(df_wide, columns, lags=[1, 3, 6, 12, 24]):
    """Create lag features for time series prediction"""
    logger.info(f"Creating lag features for {len(columns)} columns...")
    
    # Sort by location and datetime
    df_wide = df_wide.sort_values(['location_id', 'datetime_utc'])
    
    for col in columns:
        if col in df_wide.columns:
            for lag in lags:
                df_wide[f'{col}_lag_{lag}h'] = df_wide.groupby('location_id')[col].shift(lag)
    
    logger.info(f"Created lag features for lags: {lags}")
    return df_wide


def create_rolling_features(df_wide, columns, windows=[3, 6, 12, 24]):
    """Create rolling average features"""
    logger.info(f"Creating rolling features for {len(columns)} columns...")
    
    for col in columns:
        if col in df_wide.columns:
            for window in windows:
                df_wide[f'{col}_rolling_{window}h'] = df_wide.groupby('location_id')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
    
    logger.info(f"Created rolling features for windows: {windows}")
    return df_wide


def add_time_features(df_wide):
    """Add additional time-based features"""
    logger.info("Adding time-based features...")
    
    # Hour of day categories
    def get_time_of_day(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df_wide['time_of_day'] = df_wide['hour'].apply(get_time_of_day)
    
    # Rush hour flag
    df_wide['is_rush_hour'] = df_wide['hour'].apply(lambda x: 1 if x in [7, 8, 9, 17, 18, 19] else 0)
    
    return df_wide


def print_summary(df_wide):
    """Print summary of engineered dataset"""
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total records: {len(df_wide):,}")
    logger.info(f"Total features: {len(df_wide.columns)}")
    logger.info(f"Date range: {df_wide['datetime_utc'].min()} to {df_wide['datetime_utc'].max()}")
    
    logger.info(f"\nLocations: {df_wide['location_id'].nunique()}")
    for loc, count in df_wide['location_name'].value_counts().items():
        logger.info(f"  {loc}: {count:,} records")
    
    logger.info(f"\nAQI Statistics:")
    logger.info(f"  Mean: {df_wide['AQI'].mean():.1f}")
    logger.info(f"  Median: {df_wide['AQI'].median():.1f}")
    logger.info(f"  Std: {df_wide['AQI'].std():.1f}")
    logger.info(f"  Min: {df_wide['AQI'].min():.0f}")
    logger.info(f"  Max: {df_wide['AQI'].max():.0f}")
    
    logger.info(f"\nMissing values:")
    missing = df_wide.isnull().sum()
    missing_pct = (missing / len(df_wide) * 100)
    for col in missing[missing > 0].index:
        logger.info(f"  {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")
    
    logger.info("=" * 80)


def main():
    """Main feature engineering pipeline"""
    logger.info("=" * 80)
    logger.info("Starting Feature Engineering & AQI Calculation")
    logger.info("=" * 80)
    
    # Load data
    df = load_cleaned_data()
    
    # Pivot to wide format
    df_wide = pivot_to_wide_format(df)
    
    # Calculate AQI
    df_wide = calculate_aqi(df_wide)
    
    # Create lag features (for key pollutants and AQI)
    lag_columns = ['pm25', 'pm10', 'no2', 'AQI']
    df_wide = create_lag_features(df_wide, lag_columns, lags=[1, 3, 6, 12, 24])
    
    # Create rolling features
    rolling_columns = ['pm25', 'pm10', 'no2', 'AQI']
    df_wide = create_rolling_features(df_wide, rolling_columns, windows=[3, 6, 12, 24])
    
    # Add time features
    df_wide = add_time_features(df_wide)
    
    # Print summary
    print_summary(df_wide)
    
    # Save engineered dataset
    script_dir = Path(__file__).parent
    output_file = script_dir.parent / 'data' / 'processed' / 'ml_ready_data.csv'
    logger.info(f"\nSaving ML-ready dataset to {output_file}...")
    df_wide.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df_wide):,} records with {len(df_wide.columns)} features")
    
    # Save a summary for R analysis
    summary_file = script_dir.parent / 'data' / 'processed' / 'feature_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Feature Engineering Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total records: {len(df_wide):,}\n")
        f.write(f"Total features: {len(df_wide.columns)}\n")
        f.write(f"Date range: {df_wide['datetime_utc'].min()} to {df_wide['datetime_utc'].max()}\n")
        f.write(f"\nFeature categories:\n")
        f.write(f"  - Original pollutants: 7 (pm25, pm10, no2, so2, co, o3, temperature)\n")
        f.write(f"  - AQI values: 8 (AQI + 6 sub-indices + category + dominant pollutant)\n")
        f.write(f"  - Lag features: {len([c for c in df_wide.columns if 'lag' in c])}\n")
        f.write(f"  - Rolling features: {len([c for c in df_wide.columns if 'rolling' in c])}\n")
        f.write(f"  - Time features: {len([c for c in df_wide.columns if c in ['hour', 'day_of_week', 'month', 'season', 'is_weekend', 'time_of_day', 'is_rush_hour']])}\n")
    
    logger.info(f"Saved feature summary to {summary_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Feature Engineering Complete!")
    logger.info("=" * 80)
    logger.info("Next steps:")
    logger.info("  1. Run R script for statistical analysis (5_statistical_analysis.R)")
    logger.info("  2. Train ML models (6_train_models.py)")


if __name__ == "__main__":
    main()
