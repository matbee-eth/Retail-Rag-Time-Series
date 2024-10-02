# training_pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from custom_transformers import FeatureEngineer, TextPreprocessor
from sklearn.linear_model import LinearRegression
import joblib
# predict.py

import joblib
import pandas as pd
import logging
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_pipeline(pipeline_path):
    """
    Load the trained machine learning pipeline.
    
    Parameters:
    - pipeline_path (str): Path to the saved pipeline file.
    
    Returns:
    - pipeline: Loaded machine learning pipeline.
    """
    try:
        pipeline = joblib.load(pipeline_path)
        logging.info("Pipeline loaded successfully.")
        return pipeline
    except Exception as e:
        logging.error(f"Error loading the pipeline: {e}")
        raise

def prepare_input(data):
    """
    Prepare input data for prediction by converting it into a DataFrame.
    
    Parameters:
    - data (dict): Dictionary containing input features.
    
    Returns:
    - df_input (pd.DataFrame): DataFrame formatted for prediction.
    """
    df_input = pd.DataFrame([data])
    
    # Ensure datetime fields are in datetime format
    for date_col in ['Date In']:
        if date_col in df_input.columns:
            try:
                df_input[date_col] = pd.to_datetime(df_input[date_col])
            except Exception as e:
                logging.error(f"Error converting '{date_col}' to datetime: {e}")
                raise
    
    return df_input

def make_prediction(pipeline, df_input):
    """
    Make a prediction using the loaded pipeline.
    
    Parameters:
    - pipeline: Loaded machine learning pipeline.
    - df_input (pd.DataFrame): Raw input DataFrame.
    
    Returns:
    - prediction (float): Predicted Repair Duration.
    """
    try:
        prediction = pipeline.predict(df_input)[0]
        logging.info(f"Prediction successful: {prediction} days")
        return prediction
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

def main():
    # Path to the trained pipeline
    pipeline_path = 'linear_regression_model.pkl'  # Update this path if different
    
    # Load the pipeline
    pipeline = load_pipeline(pipeline_path)
    
    # Define new input data
    # **Important:** Replace the following sample data with actual data as per your schema
    new_data = {
        'Date In': '2024-10-05',  # Date repair started
        'Description': 'Replaced faulty wiring in the main control panel.',
        'Cleaned Work Order Internal Note': 'All components tested and functioning properly post-replacement.',
        'Brand and model': 'Dualtron Achilleus',
        'Type': 'E-Scooter',
        'Shop': 'Repair and Run (101) - Queen St W',
    }
    
    # Prepare the input DataFrame
    df_input = prepare_input(new_data)
    
    # Check for missing required features
    required_features = [
        'Date In', 'Description', 'Cleaned Work Order Internal Note',
        'Brand and model', 'Type', 'Shop'
    ]
    
    missing_features = [feature for feature in required_features if feature not in df_input.columns]
    if missing_features:
        raise KeyError(f"The following required features are missing from the input data: {missing_features}")
    
    # Make a prediction
    predicted_duration = make_prediction(pipeline, df_input)
    
    # Output the prediction
    print(f"\nPredicted Repair Duration: {predicted_duration:.4f} days")

if __name__ == "__main__":
    main()
