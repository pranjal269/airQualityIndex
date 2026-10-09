Air Quality Index (AQI) Prediction using Machine Learning

A comprehensive machine learning pipeline for predicting Air Quality Index (AQI) in Indian cities using real-time data from OpenAQ API. This project demonstrates end-to-end data science workflow including data collection, preprocessing, statistical analysis, feature engineering, and predictive modeling.

================================================================================

Table of Contents

1. Project Overview
2. Dataset Information
3. Project Structure
4. Installation
5. Execution Steps
6. Technical Details
7. Model Performance
8. Docker Deployment

================================================================================

Project Overview

Objective
Develop a machine learning system to predict Air Quality Index (AQI) for multiple locations in Indian cities using historical air quality data and meteorological parameters.

Key Features
- Automated data collection from OpenAQ API v3
- Comprehensive data preprocessing and cleaning pipeline
- Statistical analysis using R programming
- Advanced feature engineering with lag and rolling features
- Multiple ML models (Random Forest and XGBoost)
- Location-specific model training for improved accuracy
- Interactive Power BI dashboard for visualization

Technologies Used
- Languages: Python 3.8+, R 4.0+
- ML Libraries: scikit-learn, XGBoost
- Data Processing: pandas, numpy
- Statistical Analysis: tidyverse, ggplot2, corrplot
- Visualization: Power BI Desktop
- Deployment: Docker

================================================================================

Dataset Information

Data Source
OpenAQ API v3 - Open-source air quality data platform

Dataset Statistics
- Total Records: 29,516 (cleaned)
- Date Range: April 1 - May 28, 2025 (57 days)
- Locations: 4 monitoring stations
  * R K Puram, Delhi - DPCC
  * Punjabi Bagh, Delhi - DPCC
  * Anand Vihar, New Delhi - DPCC
  * Zoo Park, Hyderabad - TSPCB
- Parameters Measured: PM2.5, PM10, NO2, SO2, CO, O3, Temperature
- ML-Ready Records: 4,677 with 65 engineered features

Data Quality
- No missing values (100% complete)
- Outliers removed using IQR method
- Negative values eliminated
- Temporal consistency validated

================================================================================

Project Structure

project/
├── README.md                          (Project documentation)
├── requirements.txt                   (Python dependencies)
├── Dockerfile                         (Docker configuration)
├── .env                              (Environment variables - API keys)
├── .gitignore                        (Git ignore rules)
│
├── scripts/                          (Execution scripts - run in order)
│   ├── 0_test_connection.py         (API connection test)
│   ├── 1_fetch_data.py              (Data collection from OpenAQ)
│   ├── 2_preprocess_data.py         (Data cleaning and preprocessing)
│   ├── 3_inspect_data.py            (Data validation and inspection)
│   ├── 4_feature_engineering.py     (Feature creation and AQI calculation)
│   ├── 5_statistical_analysis.R     (Statistical analysis using R)
│   ├── 6_train_models.py            (ML model training)
│   └── 7_generate_predictions.py    (Generate AQI predictions)
│
├── data/                             (Data directory)
│   ├── raw/                          (Raw fetched data)
│   │   └── openaq_raw_*.csv
│   ├── processed/                    (Processed data)
│   │   ├── cleaned_data_final.csv
│   │   ├── ml_ready_data.csv
│   │   └── r_analysis/              (R analysis outputs)
│   └── predictions/                  (Model predictions)
│       └── predictions_all.csv
│
├── models/                           (Trained ML models)
│   ├── rf_*.pkl                     (Random Forest models)
│   ├── xgb_*.pkl                    (XGBoost models)
│   ├── model_metrics.csv            (Performance metrics)
│   └── feature_importance/          (Feature importance data)
│
└── AQI_Dashboard.pbix               (Power BI dashboard)

================================================================================

Installation

Prerequisites
- Python 3.8 or higher
- R 4.0 or higher (for statistical analysis)
- OpenAQ API key (free registration at https://openaq.org/)
- Power BI Desktop (for dashboard visualization)

Step 1: Clone Repository

git clone <repository-url>
cd <project-directory>

Step 2: Install Python Dependencies

pip install -r requirements.txt

Required packages:
- requests (API communication)
- python-dotenv (environment management)
- pandas (data manipulation)
- numpy (numerical operations)
- scikit-learn (machine learning)
- xgboost (gradient boosting)
- joblib (model serialization)

Step 3: Install R Packages

install.packages(c("tidyverse", "corrplot", "ggplot2", "lubridate", "gridExtra"))

Step 4: Configure Environment
Create or edit .env file with your API credentials:

OPENAQ_API_KEY=your_api_key_here
OPENAQ_API_BASE_URL=https://api.openaq.org/v3
DATE_FROM=2026-01-01
DATE_TO=2026-04-01
TARGET_CITIES=Delhi,Hyderabad
TARGET_PARAMETERS=pm25,pm10,no2,so2,co,o3,temperature

================================================================================

Execution Steps

Complete Pipeline Execution

Run scripts sequentially from the project root directory:

Script 0: Test API Connection

python scripts/0_test_connection.py

Validates API credentials and connectivity.

Script 1: Fetch Data

python scripts/1_fetch_data.py

- Collects air quality data from OpenAQ API
- Duration: 10-15 minutes
- Output: data/raw/openaq_raw_YYYYMMDD_HHMMSS.csv

Script 2: Preprocess Data

python scripts/2_preprocess_data.py

- Cleans and filters data
- Removes outliers using IQR method
- Duration: 1-2 minutes
- Output: data/processed/cleaned_data_final.csv

Script 3: Inspect Data

python scripts/3_inspect_data.py

- Validates data quality
- Displays summary statistics
- Duration: < 1 minute

Script 4: Feature Engineering

python scripts/4_feature_engineering.py

- Calculates AQI using Indian CPCB formula
- Creates lag features (1h, 3h, 6h, 12h, 24h)
- Creates rolling averages (3h, 6h, 12h, 24h)
- Adds temporal features
- Duration: 2-3 minutes
- Output: data/processed/ml_ready_data.csv

Script 5: Statistical Analysis (R)

Rscript scripts/5_statistical_analysis.R

- Performs descriptive statistics
- Generates correlation matrices
- Conducts ANOVA tests
- Creates visualizations
- Duration: 2-3 minutes
- Output: 11 files in data/processed/r_analysis/

Script 6: Train Models

python scripts/6_train_models.py

- Trains Random Forest and XGBoost models
- Performs 80-20 train-test split
- Evaluates model performance
- Duration: 3-5 minutes
- Output: 8 models in models/ directory

Script 7: Generate Predictions

python scripts/7_generate_predictions.py

- Loads trained models
- Generates AQI predictions
- Calculates prediction errors
- Duration: 1-2 minutes
- Output: data/predictions/predictions_all.csv

================================================================================

Technical Details

Data Preprocessing

Cleaning Steps:
1. Date filtering (April-May 2025)
2. Negative value removal (4 records)
3. Outlier removal using IQR method (359 records)
4. Missing value handling (forward/backward fill)
5. City name extraction from location data

Outlier Detection:
- Method: Interquartile Range (IQR)
- Formula: Q1 - 1.5×IQR to Q3 + 1.5×IQR
- Applied to all pollutant parameters

Feature Engineering

AQI Calculation:
- Standard: Indian CPCB (Central Pollution Control Board)
- Sub-indices calculated for: PM2.5, PM10, NO2, SO2, CO, O3
- Final AQI: Maximum of all sub-indices

Lag Features:
- 1-hour, 3-hour, 6-hour, 12-hour, 24-hour historical values
- Captures temporal dependencies

Rolling Features:
- 3-hour, 6-hour, 12-hour, 24-hour moving averages
- Smooths short-term fluctuations

Temporal Features:
- Hour of day (0-23)
- Day of week (0-6)
- Month (1-12)
- Season (categorical)
- Cyclical encodings (sin/cos transformations)

Total Features: 65 (54 numeric + 11 metadata/categorical)

Statistical Analysis (R)

Analyses Performed:
1. Descriptive statistics (mean, median, std dev, quartiles)
2. Correlation analysis (pollutant relationships)
3. ANOVA tests (location-wise differences)
4. Temporal pattern analysis (hourly, weekly)
5. Pollutant distribution analysis

Visualizations Generated:
- Correlation matrix
- Time series plots
- AQI by location comparison
- Category distribution charts
- Hourly and weekly pattern plots
- Pollutant distribution histograms

Machine Learning Models

Model 1: Random Forest
- Type: Ensemble learning (bagging)
- Hyperparameters:
  * n_estimators: 100
  * max_depth: 20
  * min_samples_split: 5
  * random_state: 42

Model 2: XGBoost
- Type: Gradient boosting
- Hyperparameters:
  * n_estimators: 100
  * max_depth: 6
  * learning_rate: 0.1
  * random_state: 42

Training Strategy:
- Location-specific models (4 locations)
- 80-20 chronological train-test split
- 54 numeric features used
- Missing values: forward fill + backward fill

Evaluation Metrics:
- R² (R-squared): Variance explained
- MAE (Mean Absolute Error): Average prediction error
- RMSE (Root Mean Squared Error): Penalized large errors

================================================================================

Model Performance

Overall Results

Model              | Average R² | Average MAE | Average RMSE
-------------------|-----------|-------------|-------------
Random Forest      | 0.802     | 30.65       | 47.51
XGBoost            | 0.820     | 28.15       | 45.63

Location-wise Performance

Location                    | Model         | R²    | MAE   | RMSE
----------------------------|---------------|-------|-------|-------
Zoo Park, Hyderabad         | XGBoost       | 0.932 | 14.12 | 23.77
Zoo Park, Hyderabad         | Random Forest | 0.932 | 14.98 | 23.91
Punjabi Bagh, Delhi         | Random Forest | 0.912 | 15.05 | 26.18
Punjabi Bagh, Delhi         | XGBoost       | 0.838 | 22.34 | 35.52
R K Puram, Delhi            | XGBoost       | 0.815 | 19.21 | 47.77
R K Puram, Delhi            | Random Forest | 0.773 | 25.46 | 52.82
Anand Vihar, Delhi          | XGBoost       | 0.695 | 57.14 | 75.49
Anand Vihar, Delhi          | Random Forest | 0.592 | 67.14 | 87.32

Key Insights

Best Performing Location:
- Zoo Park, Hyderabad (R² = 0.932)
- Reason: Stable air quality patterns, lower pollution variability

Most Challenging Location:
- Anand Vihar, Delhi (R² = 0.592-0.695)
- Reason: High traffic density, significant pollution variability

Model Comparison:
- XGBoost outperforms Random Forest in 3 out of 4 locations
- XGBoost better at capturing non-linear patterns
- Random Forest excellent for Punjabi Bagh (R² = 0.912)

Top Predictive Features:
1. PM2.5 (current value)
2. PM10 (current value)
3. PM2.5 24-hour lag
4. PM2.5 24-hour rolling average
5. NO2 (current value)

================================================================================

## Docker Deployment

The project is containerized using Docker. The published image includes the dependencies, processed dataset, trained models, and scripts required to generate AQI predictions.

### Pull the published image

```bash
docker pull pranjal269/air-quality-index:latest
```

### Run the prediction pipeline

```bash
docker run --rm pranjal269/air-quality-index:latest
```

The container loads the trained Random Forest and XGBoost models, processes the included ML-ready dataset, and generates predictions for four monitoring locations.

### Build the image locally

```bash
git clone https://github.com/pranjal269/airQualityIndex.git
cd airQualityIndex
docker build -t air-quality-index:latest .
docker run --rm air-quality-index:latest
```

### Output

The pipeline generates predictions at:

`data/predictions/predictions_all.csv`

**Note:** The container runs the prediction-generation pipeline using the included processed data and trained models. Running the full data-collection, R statistical-analysis, or Power BI workflows may require additional setup.

================================================================================

Configuration Reference

Environment Variables

Variable              | Description                      | Default
----------------------|----------------------------------|---------------------------
OPENAQ_API_KEY        | OpenAQ API authentication key    | Required
OPENAQ_API_BASE_URL   | API endpoint URL                 | https://api.openaq.org/v3
DATE_FROM             | Data collection start date       | 2026-01-01
DATE_TO               | Data collection end date         | 2026-04-01
TARGET_CITIES         | Comma-separated city list        | Delhi,Hyderabad
TARGET_PARAMETERS     | Pollutants to fetch              | pm25,pm10,no2,so2,co,o3,temperature
MAX_RETRIES           | API request retry attempts       | 3
TIMEOUT               | API request timeout (seconds)    | 30
REQUEST_DELAY         | Delay between requests (seconds) | 0.6

================================================================================

Troubleshooting

Common Issues

Issue: API Connection Failed
- Verify API key in .env file
- Check internet connectivity
- Confirm OpenAQ API status

Issue: R Script Fails
- Ensure R is installed and in system PATH
- Install required R packages
- Check R version (4.0+)

Issue: Model Training Memory Error
- Reduce dataset size
- Decrease model complexity (n_estimators, max_depth)
- Increase system RAM allocation

Issue: Missing Data Files
- Run scripts in sequential order (0-7)
- Check output directories exist
- Verify previous script completed successfully

================================================================================

Project Validation

To verify complete pipeline execution:

python scripts/validate_pipeline.py

This script checks:
- Data file existence and integrity
- Model file availability
- Prediction file validity
- Overall pipeline health

================================================================================

Future Enhancements

1. Real-time prediction API
2. Deep learning models (LSTM, GRU)
3. Multi-step ahead forecasting
4. Weather data integration
5. Mobile application development
6. Automated model retraining pipeline

================================================================================

Contributors

This project was developed as part of an academic data science initiative.

================================================================================

Acknowledgments

- OpenAQ for providing open-source air quality data
- Central Pollution Control Board (CPCB), India
- State Pollution Control Boards (Delhi DPCC, Telangana TSPCB)

================================================================================

License

This project is for educational and research purposes.

================================================================================

Contact

For questions or collaboration opportunities, please refer to the project repository.

================================================================================

Last Updated: April 15, 2026
Version: 1.0.0
Status: Production Ready
