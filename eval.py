# eval.py

import numpy as np
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
)
from custom_transformers import FeatureCounter, CastToFloat32, FeatureEngineer, TextPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def main():
    # Path configurations
    linear_reg_model_path = 'linear_regression_model.pkl'  # Path to your saved Linear Regression model
    data_path = 'your_repair_shop_data_cleaned.parquet'    # Path to your cleaned dataset
    
    # Load the trained Linear Regression pipeline
    try:
        linear_reg_pipeline = joblib.load(linear_reg_model_path)
        print("Linear Regression model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load the model: {e}")
        raise
    
    # Load the entire dataset
    try:
        df = pd.read_parquet(data_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        logging.error(f"File not found: {data_path}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while loading data: {e}")
        raise
    
    # Strip leading/trailing spaces from column names to ensure consistency
    df.columns = df.columns.str.strip()
    
    # Verify and create 'Repair_Description' if it doesn't exist
    if 'Repair_Description' not in df.columns:
        # Create 'Repair_Description' by combining 'Description' and 'Cleaned Work Order Internal Note'
        if 'Description' not in df.columns or 'Cleaned Work Order Internal Note' not in df.columns:
            raise KeyError("Required columns for 'Repair_Description' are missing.")
        
        df['Repair_Description'] = df['Description'].fillna('') + ' ' + df['Cleaned Work Order Internal Note'].fillna('')
        
        # Verify creation
        if 'Repair_Description' not in df.columns:
            raise KeyError("Failed to create 'Repair_Description' column.")
        else:
            logging.info("'Repair_Description' column successfully created.")
    
    # Ensure 'Date' and 'Date In' are datetime and sort the DataFrame
    for date_col in ['Date', 'Date In']:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            logging.info(f"Converted '{date_col}' to datetime.")
        except Exception as e:
            logging.error(f"Error converting '{date_col}' to datetime: {e}")
            raise
    
    # Calculate 'Repair Duration' if not present
    if 'Repair Duration' not in df.columns:
        df['Repair Duration'] = (df['Date'] - df['Date In']).dt.days
        df['Repair Duration'] = df['Repair Duration'].clip(lower=0)  # Ensure non-negative durations
        logging.info("Calculated 'Repair Duration'.")
    
    # List of monetary columns to convert
    monetary_columns = ['Total_y', 'HST', 'Discount', 'Subtotal']
    
    # Function to clean and convert monetary columns
    def clean_monetary_columns(df, columns):
        for col in columns:
            if col in df.columns:
                # Remove '$' and any commas, then convert to numeric
                df[col] = df[col].replace('[\$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logging.info(f"Converted '{col}' to numeric.")
            else:
                raise KeyError(f"'{col}' column is missing from the DataFrame.")
        return df
    
    # Clean and convert monetary columns
    df = clean_monetary_columns(df, monetary_columns)
    
    # Remove or impute missing target values
    df = df.dropna(subset=['Repair Duration', 'Total_y'])
    logging.info("Dropped rows with missing 'Repair Duration' or 'Total_y'.")
    
    # Handle potential outliers in 'Repair Duration' and 'Total_y'
    from scipy import stats
    
    def remove_outliers(df, column, threshold=3):
        """
        Remove outliers from a DataFrame based on Z-score threshold.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - column (str): The column name to check for outliers.
        - threshold (float): The Z-score threshold to identify outliers.

        Returns:
        - pd.DataFrame: DataFrame without outliers in the specified column.
        """
        z_scores = np.abs(stats.zscore(df[column]))
        initial_count = df.shape[0]
        df_filtered = df[z_scores < threshold]
        final_count = df_filtered.shape[0]
        logging.info(f"Removed {initial_count - final_count} outliers from '{column}'.")
        return df_filtered
    
    df = remove_outliers(df, 'Repair Duration')
    df = remove_outliers(df, 'Total_y')
    
    # Feature Engineering
    feature_engineer = FeatureEngineer()
    df = feature_engineer.transform(df)
    
    # Preprocess 'Repair_Description'
    text_preprocessor = TextPreprocessor()
    df = text_preprocessor.transform(df)
    
    # TF-IDF Vectorization for text data
    tfidf_vectorizer = TfidfVectorizer(max_features=100)
    try:
        tfidf_features = tfidf_vectorizer.fit_transform(df['Processed_Repair_Description']).toarray()
        tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        logging.info("TF-IDF vectorization completed.")
    except Exception as e:
        logging.error(f"Error during TF-IDF vectorization: {e}")
        raise
    
    # Concatenate TF-IDF features to the main DataFrame
    df = pd.concat([df, tfidf_df], axis=1)
    
    # Fill or drop NaN values after feature engineering
    df = df.bfill().ffill()
    
    # Drop original text columns to reduce dimensionality (excluding 'Repair_Description' as it's processed)
    columns_to_drop = ['Description', 'Combined Services + Parts', 'Processed_Repair_Description']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(existing_columns_to_drop, axis=1)
    
    # Define feature columns as used during training
    feature_columns = [
        'Year', 'Month', 'WeekOfYear', 'DayOfWeek', 'Quarter', 'IsWeekend',
        'Brand and model', 'Type', 'Item', 'Shop', 'Status', 'Note Author',
        'Qty', 'HST', 'Discount', 'Subtotal',
        'lag_1', 'lag_2', 'rolling_mean_3'
    ] + [f'tfidf_{i}' for i in range(100)]  # Adjust based on actual tfidf features
    
    # Verify that all feature columns exist in the dataset
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise KeyError(f"The following feature columns are missing from the dataset: {missing_features}")
    else:
        print("All required feature columns are present.")
    
    # Prepare features and target
    X = df[feature_columns]
    y = df['Repair Duration'].astype(float)  # Ensure target is float
    
    # Make predictions using the trained pipeline
    try:
        y_pred = linear_reg_pipeline.predict(X)
        print("Predictions made successfully.")
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        raise
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(y, y_pred)
    median_ae = median_absolute_error(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)  # squared=False returns RMSE
    r2 = r2_score(y, y_pred)
    explained_var = explained_variance_score(y, y_pred)
    
    # Handle MAPE: Avoid division by zero by excluding zero actual values
    non_zero_indices = y != 0
    valid_y = y[non_zero_indices]
    valid_pred = y_pred[non_zero_indices]
    
    if valid_y.shape[0] > 0:
        mape = mean_absolute_percentage_error(valid_y, valid_pred) * 100  # Convert to percentage
    else:
        mape = np.nan  # Undefined if no non-zero actual values
    
    # Display the metrics
    print("\n--- Linear Regression Evaluation Metrics on Entire Dataset ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f} days")
    print(f"Median Absolute Error (Median AE): {median_ae:.4f} days")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} days")
    print(f"R-squared: {r2:.4f}")
    print(f"Explained Variance: {explained_var:.4f}")
    if not np.isnan(mape):
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    else:
        print("Mean Absolute Percentage Error (MAPE): Undefined (No non-zero actual values)")
    
    # Visualization
    
    # Scatter plot of Actual vs Predicted Repair Duration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Diagonal line for reference
    plt.xlabel('Actual Repair Duration (days)')
    plt.ylabel('Predicted Repair Duration (days)')
    plt.title('Linear Regression: Actual vs Predicted Repair Duration')
    plt.grid(True)
    plt.show()
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Histogram of Residuals
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, bins=50, kde=True)
    plt.xlabel('Residuals (days)')
    plt.title('Residuals Distribution for Linear Regression')
    plt.grid(True)
    plt.show()
    
    # Residuals vs. Predicted Repair Duration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Repair Duration (days)')
    plt.ylabel('Residuals (days)')
    plt.title('Residuals vs Predicted Repair Duration for Linear Regression')
    plt.grid(True)
    plt.show()
    
    # Residuals vs. Actual Repair Duration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y, y=residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Actual Repair Duration (days)')
    plt.ylabel('Residuals (days)')
    plt.title('Residuals vs Actual Repair Duration for Linear Regression')
    plt.grid(True)
    plt.show()
    
    # Feature Importance Analysis
    # Access the trained Linear Regression model from the pipeline
    linear_reg = linear_reg_pipeline.named_steps['regressor']
    
    # Get feature names after preprocessing
    # Assuming TfidfVectorizer was used for 'Processed_Repair_Description'
    preprocessor = linear_reg_pipeline.named_steps['preprocessor']
    
    # Extract TF-IDF feature names
    tfidf_feature_names = [f'tfidf_{i}' for i in range(100)]  # Adjust if different
    
    # Combine numerical and TF-IDF feature names
    numerical_cols_extended = [
        'Qty', 'HST', 'Discount', 'Subtotal',
        'lag_1', 'lag_2', 'rolling_mean_3',
        'Year', 'Month', 'WeekOfYear', 'DayOfWeek', 'Quarter', 'IsWeekend'
    ] + tfidf_feature_names
    all_feature_names = numerical_cols_extended + ['Brand and model', 'Type', 'Item', 'Shop', 'Status', 'Note Author']
    
    # Get coefficients
    coefficients = linear_reg.coef_
    
    # Create a DataFrame for feature importances
    feature_importances = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefficients
    })
    
    # Absolute coefficients for importance
    feature_importances['Absolute Coefficient'] = feature_importances['Coefficient'].abs()
    
    # Sort features by importance
    feature_importances_sorted = feature_importances.sort_values(by='Absolute Coefficient', ascending=False).head(20)  # Top 20 features
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Absolute Coefficient', y='Feature', data=feature_importances_sorted)
    plt.xlabel('Absolute Coefficient')
    plt.ylabel('Feature')
    plt.title('Top 20 Feature Importances based on Linear Regression Coefficients')
    plt.tight_layout()
    plt.show()
    
    # Optional: Save evaluation metrics to a CSV file
    evaluation_metrics = {
        'MAE': [mae],
        'Median AE': [median_ae],
        'RMSE': [rmse],
        'R-squared': [r2],
        'Explained Variance': [explained_var],
        'MAPE': [mape] if not np.isnan(mape) else [np.nan]
    }
    metrics_df = pd.DataFrame(evaluation_metrics)
    metrics_df.to_csv('linear_regression_evaluation_metrics.csv', index=False)
    logging.info("Evaluation metrics saved to 'linear_regression_evaluation_metrics.csv'.")
    
    # Optional: Save feature importances to a CSV file
    feature_importances_sorted.to_csv('linear_regression_feature_importances.csv', index=False)
    logging.info("Feature importances saved to 'linear_regression_feature_importances.csv'.")
    
if __name__ == "__main__":
    main()
