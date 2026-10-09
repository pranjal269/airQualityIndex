"""
Data Preprocessing Script for AQI Prediction
- Filter data to 2021-2025
- Clean negative values and outliers
- Extract city from location names
- Handle missing values
- Save cleaned dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """Load raw data"""
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df):,} records")
    return df

def filter_by_date(df, start_date='2025-03-15', end_date='2025-05-29'):
    """Filter data to specific date range"""
    logger.info(f"Filtering data from {start_date} to {end_date}...")
    
    # Convert datetime to pandas datetime
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    
    # Convert filter dates to datetime with UTC timezone
    start_dt = pd.to_datetime(start_date).tz_localize('UTC')
    end_dt = pd.to_datetime(end_date).tz_localize('UTC')
    
    # Filter
    df_filtered = df[(df['datetime_utc'] >= start_dt) & (df['datetime_utc'] <= end_dt)].copy()
    
    logger.info(f"Filtered from {len(df):,} to {len(df_filtered):,} records")
    logger.info(f"Date range: {df_filtered['datetime_utc'].min()} to {df_filtered['datetime_utc'].max()}")
    logger.info(f"Duration: {(df_filtered['datetime_utc'].max() - df_filtered['datetime_utc'].min()).days} days")
    
    return df_filtered

def extract_city(df):
    """Extract city name from location_name"""
    logger.info("Extracting city from location names...")
    
    # Extract city from pattern: "Location, City - Authority"
    df['city'] = df['location_name'].str.extract(r',\s*([^-]+)\s*-')[0]
    
    # Clean up city names
    df['city'] = df['city'].str.strip()
    
    # Handle special cases
    df.loc[df['location_name'].str.contains('New Delhi', case=False, na=False), 'city'] = 'Delhi'
    
    logger.info(f"Cities found: {df['city'].unique()}")
    logger.info(f"Records per city:\n{df['city'].value_counts()}")
    
    return df

def clean_negative_values(df):
    """Remove negative values (sensor errors)"""
    logger.info("Cleaning negative values...")
    
    initial_count = len(df)
    df_clean = df[df['value'] >= 0].copy()
    removed = initial_count - len(df_clean)
    
    logger.info(f"Removed {removed:,} records with negative values ({removed/initial_count*100:.2f}%)")
    
    return df_clean

def remove_outliers(df, method='iqr', multiplier=3):
    """Remove outliers using IQR method"""
    logger.info(f"Removing outliers using {method} method...")
    
    initial_count = len(df)
    df_clean = df.copy()
    
    for param in df['parameter'].unique():
        param_data = df_clean[df_clean['parameter'] == param]['value']
        
        if method == 'iqr':
            Q1 = param_data.quantile(0.25)
            Q3 = param_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
        elif method == 'percentile':
            lower_bound = param_data.quantile(0.001)
            upper_bound = param_data.quantile(0.999)
        
        # Filter outliers
        mask = (df_clean['parameter'] == param) & \
               ((df_clean['value'] < lower_bound) | (df_clean['value'] > upper_bound))
        
        outliers_count = mask.sum()
        logger.info(f"  {param}: Removed {outliers_count:,} outliers (range: {lower_bound:.2f} - {upper_bound:.2f})")
        
        df_clean = df_clean[~mask]
    
    removed = initial_count - len(df_clean)
    logger.info(f"Total outliers removed: {removed:,} ({removed/initial_count*100:.2f}%)")
    
    return df_clean

def add_temporal_features(df):
    """Add temporal features for ML"""
    logger.info("Adding temporal features...")
    
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    
    # Extract temporal components
    df['hour'] = df['datetime_utc'].dt.hour
    df['day_of_week'] = df['datetime_utc'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_month'] = df['datetime_utc'].dt.day
    df['month'] = df['datetime_utc'].dt.month
    df['year'] = df['datetime_utc'].dt.year
    df['day_of_year'] = df['datetime_utc'].dt.dayofyear
    
    # Cyclical encoding for hour (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for month (12-month cycle)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Season (Indian context: Winter, Summer, Monsoon, Post-Monsoon)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Summer'
        elif month in [6, 7, 8, 9]:
            return 'Monsoon'
        else:
            return 'Post-Monsoon'
    
    df['season'] = df['month'].apply(get_season)
    
    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    logger.info("Temporal features added successfully")
    
    return df

def print_summary(df):
    """Print data summary"""
    logger.info("\n" + "=" * 80)
    logger.info("CLEANED DATA SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total records: {len(df):,}")
    logger.info(f"Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
    logger.info(f"Duration: {(df['datetime_utc'].max() - df['datetime_utc'].min()).days} days")
    
    logger.info(f"\nCities ({df['city'].nunique()}):")
    for city, count in df['city'].value_counts().items():
        logger.info(f"  {city}: {count:,} records ({count/len(df)*100:.2f}%)")
    
    logger.info(f"\nParameters ({df['parameter'].nunique()}):")
    for param, count in df['parameter'].value_counts().items():
        logger.info(f"  {param}: {count:,} records ({count/len(df)*100:.2f}%)")
    
    logger.info(f"\nLocations: {df['location_id'].nunique()}")
    logger.info(f"Sensors: {df['sensor_id'].nunique()}")
    
    logger.info(f"\nValue statistics by parameter:")
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param]['value']
        logger.info(f"  {param}:")
        logger.info(f"    Min: {param_data.min():.2f}, Max: {param_data.max():.2f}")
        logger.info(f"    Mean: {param_data.mean():.2f}, Median: {param_data.median():.2f}")
        logger.info(f"    Std: {param_data.std():.2f}")
    
    logger.info(f"\nMissing values:")
    missing = df.isnull().sum()
    for col in missing[missing > 0].index:
        logger.info(f"  {col}: {missing[col]:,} ({missing[col]/len(df)*100:.2f}%)")
    
    logger.info("=" * 80)

def main():
    """Main preprocessing pipeline"""
    logger.info("=" * 80)
    logger.info("Starting Data Preprocessing")
    logger.info("=" * 80)
    
    # Load data
    df = load_data('../data/raw/openaq_raw_20260415_090246.csv')
    
    # Filter by date (April 1 - May 29, 2025 ~ 2 months for ~30k records)
    df = filter_by_date(df, start_date='2025-04-01', end_date='2025-05-29')
    
    # Extract city
    df = extract_city(df)
    
    # Clean negative values
    df = clean_negative_values(df)
    
    # Remove outliers
    df = remove_outliers(df, method='iqr', multiplier=3)
    
    # Add temporal features
    df = add_temporal_features(df)
    
    # Print summary
    print_summary(df)
    
    # Save cleaned data
    output_file = '../data/processed/cleaned_data_final.csv'
    logger.info(f"\nSaving cleaned data to {output_file}...")
    
    # Create directory if it doesn't exist
    import os
    os.makedirs('../data/processed', exist_ok=True)
    
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df):,} records to {output_file}")
    
    # Save summary statistics
    summary_file = '../data/processed/cleaning_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Data Cleaning Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Original records: 141,091\n")
        f.write(f"Filtered to 2021-2025: {len(df):,}\n")
        f.write(f"Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}\n")
        f.write(f"\nCities: {', '.join(df['city'].unique())}\n")
        f.write(f"Parameters: {', '.join(df['parameter'].unique())}\n")
        f.write(f"Locations: {df['location_id'].nunique()}\n")
        f.write(f"Sensors: {df['sensor_id'].nunique()}\n")
    
    logger.info(f"Saved summary to {summary_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 80)
    logger.info(f"Next steps:")
    logger.info(f"  1. Review cleaned data: {output_file}")
    logger.info(f"  2. Pivot to wide format (parameters as columns)")
    logger.info(f"  3. Calculate AQI values")
    logger.info(f"  4. Create lag and rolling features")
    logger.info(f"  5. Train-test split")

if __name__ == "__main__":
    main()
