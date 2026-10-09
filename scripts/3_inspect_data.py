"""
Quick data inspection script
Run after data collection to validate and explore the dataset
"""

import pandas as pd
import glob
import os
from pathlib import Path

def find_latest_data():
    """Find the most recent data file"""
    data_files = glob.glob('../data/raw/openaq_raw_*.csv')
    if not data_files:
        print("No data files found in ../data/raw/")
        return None
    return max(data_files, key=os.path.getctime)

def inspect_data(file_path):
    """Comprehensive data inspection"""
    print("=" * 80)
    print(f"Inspecting: {file_path}")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Basic info
    print(f"\n📊 BASIC STATISTICS")
    print(f"{'─' * 80}")
    print(f"Total records: {len(df):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Columns: {len(df.columns)}")
    
    # Date range
    print(f"\n📅 DATE RANGE")
    print(f"{'─' * 80}")
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    print(f"Start: {df['datetime_utc'].min()}")
    print(f"End: {df['datetime_utc'].max()}")
    print(f"Duration: {(df['datetime_utc'].max() - df['datetime_utc'].min()).days} days")
    
    # Cities
    print(f"\n🏙️  CITIES ({df['city'].nunique()} unique)")
    print(f"{'─' * 80}")
    city_counts = df['city'].value_counts()
    for city, count in city_counts.items():
        print(f"  {city:20s}: {count:6,} records ({count/len(df)*100:5.2f}%)")
    
    # Parameters
    print(f"\n🌡️  PARAMETERS ({df['parameter'].nunique()} unique)")
    print(f"{'─' * 80}")
    param_counts = df['parameter'].value_counts()
    for param, count in param_counts.items():
        print(f"  {param:15s}: {count:6,} records ({count/len(df)*100:5.2f}%)")
    
    # Data quality
    print(f"\n✓ DATA QUALITY")
    print(f"{'─' * 80}")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    print(f"Missing values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col:20s}: {missing[col]:6,} ({missing_pct[col]:5.2f}%)")
    
    if missing.sum() == 0:
        print("  ✓ No missing values!")
    
    # Value statistics for numeric parameters
    print(f"\n📈 VALUE STATISTICS")
    print(f"{'─' * 80}")
    
    for param in df['parameter'].unique():
        param_data = df[df['parameter'] == param]['value']
        print(f"\n  {param.upper()}:")
        print(f"    Count: {len(param_data):,}")
        print(f"    Mean: {param_data.mean():.3f}")
        print(f"    Std: {param_data.std():.3f}")
        print(f"    Min: {param_data.min():.3f}")
        print(f"    25%: {param_data.quantile(0.25):.3f}")
        print(f"    50%: {param_data.quantile(0.50):.3f}")
        print(f"    75%: {param_data.quantile(0.75):.3f}")
        print(f"    Max: {param_data.max():.3f}")
    
    # Coverage statistics
    print(f"\n📊 COVERAGE STATISTICS")
    print(f"{'─' * 80}")
    if 'coverage_percent' in df.columns:
        coverage = df['coverage_percent'].dropna()
        print(f"  Mean coverage: {coverage.mean():.2f}%")
        print(f"  Median coverage: {coverage.median():.2f}%")
        print(f"  Min coverage: {coverage.min():.2f}%")
        print(f"  Records with >90% coverage: {(coverage > 90).sum():,} ({(coverage > 90).sum()/len(coverage)*100:.2f}%)")
    
    # Unique sensors and locations
    print(f"\n🔍 MONITORING INFRASTRUCTURE")
    print(f"{'─' * 80}")
    print(f"  Unique sensors: {df['sensor_id'].nunique()}")
    print(f"  Unique locations: {df['location_id'].nunique()}")
    print(f"  Avg sensors per location: {df['sensor_id'].nunique() / df['location_id'].nunique():.2f}")
    
    # Temporal distribution
    print(f"\n⏰ TEMPORAL DISTRIBUTION")
    print(f"{'─' * 80}")
    df['hour'] = df['datetime_utc'].dt.hour
    df['day_of_week'] = df['datetime_utc'].dt.day_name()
    df['month'] = df['datetime_utc'].dt.month_name()
    
    print(f"  Records by month:")
    for month, count in df['month'].value_counts().sort_index().items():
        print(f"    {month:15s}: {count:6,}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print(f"{'─' * 80}")
    
    if len(df) < 25000:
        print("  ⚠ Record count below target (30,000). Consider:")
        print("    - Extending date range")
        print("    - Adding more cities")
        print("    - Using daily aggregation for more coverage")
    elif len(df) > 35000:
        print("  ✓ Excellent! Record count exceeds target.")
    else:
        print("  ✓ Record count within target range (25k-35k)")
    
    if df['parameter'].nunique() < 6:
        print("  ⚠ Limited parameters. Missing key pollutants for AQI calculation.")
    else:
        print("  ✓ Good parameter coverage for AQI prediction")
    
    if df['city'].nunique() < 12:
        print("  ⚠ Limited city coverage. Consider adding more cities.")
    else:
        print("  ✓ Good geographic diversity")
    
    # Check for essential parameters
    essential_params = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
    available_params = df['parameter'].unique()
    missing_params = [p for p in essential_params if p not in available_params]
    
    if missing_params:
        print(f"  ⚠ Missing essential parameters: {', '.join(missing_params)}")
    else:
        print("  ✓ All essential AQI parameters present")
    
    print("\n" + "=" * 80)
    print("Inspection complete!")
    print("=" * 80)
    
    return df

def main():
    """Main execution"""
    file_path = find_latest_data()
    
    if file_path:
        df = inspect_data(file_path)
        
        # Ask if user wants to save summary
        print("\n💾 Save detailed summary to file? (y/n): ", end='')
        response = input().strip().lower()
        
        if response == 'y':
            summary_file = file_path.replace('.csv', '_summary.txt')
            import sys
            original_stdout = sys.stdout
            with open(summary_file, 'w') as f:
                sys.stdout = f
                inspect_data(file_path)
            sys.stdout = original_stdout
            print(f"✓ Summary saved to: {summary_file}")
    else:
        print("No data files found. Please run fetch_openaq_data.py first.")

if __name__ == "__main__":
    main()
