"""
Quick test script to verify OpenAQ API connection and data availability
Run this before the full data fetch to ensure everything is configured correctly
"""

import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

def test_api_connection():
    """Test basic API connectivity"""
    api_key = os.getenv('OPENAQ_API_KEY')
    base_url = os.getenv('OPENAQ_API_BASE_URL', 'https://api.openaq.org/v3')
    
    headers = {
        'X-API-Key': api_key,
        'Accept': 'application/json'
    }
    
    print("=" * 80)
    print("OpenAQ API Connection Test")
    print("=" * 80)
    print(f"API Key: {api_key[:20]}..." if api_key else "API Key: NOT FOUND")
    print(f"Base URL: {base_url}")
    print()
    
    # Test 1: Get countries
    print("Test 1: Fetching India country info...")
    try:
        response = requests.get(
            f"{base_url}/countries/9",  # India's country ID in OpenAQ v3
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        result = data.get('results', [{}])
        if isinstance(result, list):
            result = result[0] if result else {}
        print(f"✓ Success! Found: {result.get('name', 'N/A')}")
        print(f"  Country code: {result.get('code', 'N/A')}")
        print(f"  Total locations: {result.get('locations', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print()
    
    # Test 2: Get Delhi locations
    print("Test 2: Fetching Delhi locations...")
    try:
        response = requests.get(
            f"{base_url}/locations",
            headers=headers,
            params={'country_id': 9, 'limit': 5},  # India's country_id
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        print(f"✓ Success! Found {len(results)} locations")
        
        if results:
            location = results[0]
            print(f"\n  Sample Location:")
            print(f"    Name: {location.get('name')}")
            print(f"    ID: {location.get('id')}")
            print(f"    Locality: {location.get('locality', 'N/A')}")
            print(f"    Sensors: {len(location.get('sensors', []))}")
            
            # Show available parameters
            params = [s.get('parameter', {}).get('name') for s in location.get('sensors', [])]
            print(f"    Parameters: {', '.join(params)}")
            
            # Show date range - handle None values
            datetime_first = location.get('datetimeFirst', {})
            datetime_last = location.get('datetimeLast', {})
            if datetime_first:
                print(f"    First measurement: {datetime_first.get('local', 'N/A')}")
            if datetime_last:
                print(f"    Last measurement: {datetime_last.get('local', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print()
    
    # Test 3: Get sample measurements
    if results and results[0].get('sensors'):
        sensor_id = results[0]['sensors'][0]['id']
        print(f"Test 3: Fetching sample measurements for sensor {sensor_id}...")
        try:
            response = requests.get(
                f"{base_url}/sensors/{sensor_id}/hours",
                headers=headers,
                params={
                    'date_from': '2026-04-01',
                    'date_to': '2026-04-15',
                    'limit': 5
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            measurements = data.get('results', [])
            print(f"✓ Success! Found {len(measurements)} measurements")
            
            if measurements:
                m = measurements[0]
                print(f"\n  Sample Measurement:")
                print(f"    Parameter: {m.get('parameter', {}).get('name')}")
                print(f"    Value: {m.get('value')} {m.get('parameter', {}).get('units')}")
                print(f"    DateTime: {m.get('period', {}).get('datetimeFrom', {}).get('local')}")
                print(f"    Coverage: {m.get('coverage', {}).get('percentComplete')}%")
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False
    
    print()
    print("=" * 80)
    print("✓ All tests passed! API is ready for data collection.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_api_connection()
    if not success:
        print("\n⚠ Please check your API key and internet connection before proceeding.")
        exit(1)
