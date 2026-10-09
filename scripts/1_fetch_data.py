"""
OpenAQ Data Fetcher for AQI Prediction
Fetches air quality data from OpenAQ API v3 for 15 Indian cities
Target: ~30,000 hourly records over 3 months (Jan 15 - Apr 15, 2026)
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from pathlib import Path
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_fetch.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OpenAQFetcher:
    """Fetches air quality data from OpenAQ API v3"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAQ_API_KEY')
        self.base_url = os.getenv('OPENAQ_API_BASE_URL', 'https://api.openaq.org/v3')
        self.max_retries = int(os.getenv('MAX_RETRIES', 3))
        self.timeout = int(os.getenv('TIMEOUT', 30))
        self.request_delay = float(os.getenv('REQUEST_DELAY', 0.6))
        self.fetch_limit = int(os.getenv('OPENAQ_FETCH_LIMIT', 1000))
        self.max_pages = int(os.getenv('OPENAQ_MAX_PAGES', 100))
        
        # Date range
        self.date_from = os.getenv('DATE_FROM', '2026-01-15')
        self.date_to = os.getenv('DATE_TO', '2026-04-15')
        
        # Target cities and parameters
        self.target_cities = os.getenv('TARGET_CITIES', '').split(',')
        self.target_parameters = os.getenv('TARGET_PARAMETERS', '').split(',')
        
        # Headers for API requests
        self.headers = {
            'X-API-Key': self.api_key,
            'Accept': 'application/json'
        }
        
        # Create data directory
        self.data_dir = Path('../data/raw')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized OpenAQ Fetcher")
        logger.info(f"Date range: {self.date_from} to {self.date_to}")
        logger.info(f"Target cities: {len(self.target_cities)}")
        logger.info(f"Target parameters: {self.target_parameters}")
    
    def _make_request(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                time.sleep(self.request_delay)  # Rate limiting
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {url}")
                    return None
        
        return None
    
    def get_locations_for_city(self, city_name: str) -> List[Dict]:
        """Get all monitoring locations for a specific city"""
        logger.info(f"Fetching locations for {city_name}...")
        
        url = f"{self.base_url}/locations"
        params = {
            'country_id': 9,  # India's country ID in OpenAQ v3
            'limit': self.fetch_limit,
            'page': 1
        }
        
        all_locations = []
        page = 1
        
        while page <= self.max_pages:
            params['page'] = page
            data = self._make_request(url, params)
            
            if not data or 'results' not in data:
                break
            
            results = data['results']
            if not results:
                break
            
            # Filter by city name in locality field
            city_locations = [
                loc for loc in results 
                if city_name.lower() in str(loc.get('locality', '')).lower() or
                   city_name.lower() in str(loc.get('name', '')).lower()
            ]
            
            all_locations.extend(city_locations)
            logger.info(f"  Page {page}: Found {len(city_locations)} locations for {city_name}")
            
            # Check if there are more pages
            meta = data.get('meta', {})
            found = meta.get('found', 0)
            if isinstance(found, str) and found.startswith('>'):
                # Handle ">1000" format
                found = int(found.replace('>', ''))
            if page >= found / self.fetch_limit:
                break
            
            page += 1
        
        logger.info(f"Total locations for {city_name}: {len(all_locations)}")
        return all_locations
    
    def extract_sensors_from_locations(self, locations: List[Dict], 
                                      target_params: List[str]) -> List[Dict]:
        """Extract sensor information from locations"""
        sensors_info = []
        
        for location in locations:
            location_id = location.get('id')
            location_name = location.get('name')
            city = location.get('locality', location.get('name'))
            coordinates = location.get('coordinates', {})
            
            for sensor in location.get('sensors', []):
                param_name = sensor.get('parameter', {}).get('name', '').lower()
                
                if param_name in target_params:
                    sensors_info.append({
                        'sensor_id': sensor.get('id'),
                        'location_id': location_id,
                        'location_name': location_name,
                        'city': city,
                        'parameter': param_name,
                        'units': sensor.get('parameter', {}).get('units'),
                        'latitude': coordinates.get('latitude'),
                        'longitude': coordinates.get('longitude')
                    })
        
        return sensors_info
    
    def fetch_sensor_measurements(self, sensor_id: int, sensor_info: Dict) -> pd.DataFrame:
        """Fetch hourly measurements for a specific sensor"""
        logger.info(f"Fetching measurements for sensor {sensor_id} ({sensor_info['parameter']})...")
        
        url = f"{self.base_url}/sensors/{sensor_id}/hours"
        params = {
            'date_from': self.date_from,
            'date_to': self.date_to,
            'limit': self.fetch_limit,
            'page': 1
        }
        
        all_measurements = []
        page = 1
        
        while page <= self.max_pages:
            params['page'] = page
            data = self._make_request(url, params)
            
            if not data or 'results' not in data:
                break
            
            results = data['results']
            if not results:
                break
            
            # Parse measurements
            for measurement in results:
                all_measurements.append({
                    'sensor_id': sensor_id,
                    'location_id': sensor_info['location_id'],
                    'location_name': sensor_info['location_name'],
                    'city': sensor_info['city'],
                    'parameter': sensor_info['parameter'],
                    'value': measurement.get('value'),
                    'units': sensor_info['units'],
                    'datetime_utc': measurement.get('period', {}).get('datetimeFrom', {}).get('utc'),
                    'datetime_local': measurement.get('period', {}).get('datetimeFrom', {}).get('local'),
                    'latitude': sensor_info['latitude'],
                    'longitude': sensor_info['longitude'],
                    'coverage_percent': measurement.get('coverage', {}).get('percentComplete'),
                })
            
            logger.info(f"  Page {page}: {len(results)} measurements")
            
            # Check if there are more pages
            meta = data.get('meta', {})
            found = meta.get('found', 0)
            
            # Handle ">1000" format from API
            if isinstance(found, str):
                if found.startswith('>'):
                    found = int(found.replace('>', '')) + 1
                else:
                    found = int(found)
            
            if page >= found / self.fetch_limit:
                break
            
            page += 1
        
        df = pd.DataFrame(all_measurements)
        logger.info(f"Total measurements for sensor {sensor_id}: {len(df)}")
        return df
    
    def fetch_all_data(self) -> pd.DataFrame:
        """Main method to fetch all data for all cities"""
        logger.info("=" * 80)
        logger.info("Starting OpenAQ data collection")
        logger.info("=" * 80)
        
        all_data = []
        total_sensors = 0
        
        for city in self.target_cities:
            city = city.strip()
            if not city:
                continue
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Processing city: {city}")
            logger.info(f"{'=' * 80}")
            
            # Get locations for city
            locations = self.get_locations_for_city(city)
            
            if not locations:
                logger.warning(f"No locations found for {city}")
                continue
            
            # Extract sensors
            sensors = self.extract_sensors_from_locations(locations, self.target_parameters)
            logger.info(f"Found {len(sensors)} sensors for target parameters")
            
            # Fetch measurements for each sensor
            for idx, sensor_info in enumerate(sensors, 1):
                logger.info(f"\nSensor {idx}/{len(sensors)}")
                df = self.fetch_sensor_measurements(sensor_info['sensor_id'], sensor_info)
                
                if not df.empty:
                    all_data.append(df)
                    total_sensors += 1
                
                # Save intermediate results every 10 sensors
                if total_sensors % 10 == 0:
                    self._save_intermediate_results(all_data, total_sensors)
        
        # Combine all data
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Data collection complete!")
            logger.info(f"Total records: {len(final_df)}")
            logger.info(f"Total sensors: {total_sensors}")
            logger.info(f"{'=' * 80}")
            return final_df
        else:
            logger.error("No data collected!")
            return pd.DataFrame()
    
    def _save_intermediate_results(self, data_list: List[pd.DataFrame], count: int):
        """Save intermediate results to prevent data loss"""
        if data_list:
            temp_df = pd.concat(data_list, ignore_index=True)
            temp_file = self.data_dir / f'intermediate_{count}_sensors.csv'
            temp_df.to_csv(temp_file, index=False)
            logger.info(f"Saved intermediate results: {temp_file}")
    
    def save_data(self, df: pd.DataFrame):
        """Save final dataset"""
        if df.empty:
            logger.error("No data to save!")
            return
        
        # Save raw data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_file = self.data_dir / f'openaq_raw_{timestamp}.csv'
        df.to_csv(raw_file, index=False)
        logger.info(f"Saved raw data: {raw_file}")
        
        # Save metadata
        metadata = {
            'fetch_date': timestamp,
            'date_range': f"{self.date_from} to {self.date_to}",
            'total_records': len(df),
            'cities': df['city'].nunique(),
            'parameters': df['parameter'].unique().tolist(),
            'date_range_actual': {
                'min': df['datetime_utc'].min(),
                'max': df['datetime_utc'].max()
            }
        }
        
        metadata_file = self.data_dir / f'metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_file}")
        
        # Print summary statistics
        self._print_summary(df)
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        logger.info("\n" + "=" * 80)
        logger.info("DATA SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")
        logger.info(f"\nRecords by city:")
        for city, count in df['city'].value_counts().items():
            logger.info(f"  {city}: {count}")
        logger.info(f"\nRecords by parameter:")
        for param, count in df['parameter'].value_counts().items():
            logger.info(f"  {param}: {count}")
        logger.info("=" * 80)


def main():
    """Main execution function"""
    fetcher = OpenAQFetcher()
    
    # Fetch all data
    df = fetcher.fetch_all_data()
    
    # Save data
    if not df.empty:
        fetcher.save_data(df)
    else:
        logger.error("Data collection failed - no data retrieved")


if __name__ == "__main__":
    main()
