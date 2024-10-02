import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
)
import joblib
from pathlib import Path
import logging
import warnings
from scipy import stats
import torch
from skorch import NeuralNetRegressor
from skorch.callbacks import EarlyStopping
from torch import nn, optim
from sklearn.base import TransformerMixin, BaseEstimator

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Ensure NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"GPU is available. Using device: {device}")
else:
    device = torch.device("cpu")
    logging.info("GPU not available. Using CPU.")

# Define the custom transformer to cast data to float32
class CastToFloat32(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.astype(np.float32)

# Define a custom transformer to log feature counts
class FeatureCounter(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logging.info(f"Number of features after preprocessing: {X.shape[1]}")
        return X

# Define the SafeNeuralNetRegressor subclass to address FutureWarning
class SafeNeuralNetRegressor(NeuralNetRegressor):
    def load_params(self, f_params=None, **kwargs):
        # Override to set weights_only=True
        kwargs['weights_only'] = True
        super().load_params(f_params=f_params, **kwargs)

# Define the PyTorch MLP model with modified forward method
class PyTorchMLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[150, 75], activation=nn.ReLU):
        super(PyTorchMLPRegressor, self).__init__()
        layers = []
        in_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, 1))  # Regression output
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.network(x)
        return output.squeeze(1)  # Ensures output shape is [batch_size]

# Load the cleaned data
cleaned_data_path = Path('your_repair_shop_data_cleaned.parquet')  # Update the path accordingly
try:
    df = pd.read_parquet(cleaned_data_path)
    logging.info(f"Data loaded successfully from {cleaned_data_path}")
except FileNotFoundError:
    logging.error(f"File not found: {cleaned_data_path}")
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

# Ensure 'Date In' is datetime and sort the DataFrame
try:
    df['Date In'] = pd.to_datetime(df['Date In'])
    logging.info(f"Converted 'Date In' to datetime.")
except Exception as e:
    logging.error(f"Error converting 'Date In' to datetime: {e}")
    raise

df = df.sort_values('Date In').reset_index(drop=True)

# Calculate repair duration
df['Repair Duration'] = (df['Date'] - df['Date In']).dt.days
df['Repair Duration'] = df['Repair Duration'].clip(lower=0)  # Ensure non-negative durations
logging.info("Calculated 'Repair Duration'.")

# Remove or impute missing target values
df = df.dropna(subset=['Repair Duration'])
logging.info("Dropped rows with missing 'Repair Duration'.")

# Handle potential outliers in 'Repair Duration'
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

# Feature Engineering

# Extract additional time-based features from 'Date In'
df['Year'] = df['Date In'].dt.year
df['Month'] = df['Date In'].dt.month
df['WeekOfYear'] = df['Date In'].dt.isocalendar().week
df['DayOfWeek'] = df['Date In'].dt.dayofweek
df['Quarter'] = df['Date In'].dt.quarter
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Encode categorical variables - Removed 'Item', 'Note Author', 'Status' as they are not part of input
categorical_features = ['Brand and model', 'Type', 'Shop']

# Fill missing categorical data
df[categorical_features] = df[categorical_features].fillna('Unknown')

# Preprocess 'Repair_Description'
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

df['Processed_Repair_Description'] = df['Repair_Description'].apply(preprocess_text)

# TF-IDF Vectorization for text data
tfidf_vectorizer = TfidfVectorizer(max_features=100)
tfidf_features = tfidf_vectorizer.fit_transform(df['Processed_Repair_Description']).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])

# Concatenate TF-IDF features to the main DataFrame
df = pd.concat([df, tfidf_df], axis=1)

# Handle NaN values resulting from shifting and rolling - Not needed as lag features are removed

# Drop original text columns to reduce dimensionality
columns_to_drop = ['Description', 'Cleaned Work Order Internal Note', 'Repair_Description', 'Processed_Repair_Description']
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(existing_columns_to_drop, axis=1)

# Feature Selection for Modeling
feature_columns = [
    'Year', 'Month', 'WeekOfYear', 'DayOfWeek', 'Quarter', 'IsWeekend',
    'Brand and model', 'Type', 'Shop'
] + list(tfidf_df.columns)

# Ensure all feature columns exist
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    raise KeyError(f"The following feature columns are missing from the DataFrame: {missing_features}")

X = df[feature_columns]
y_duration = df['Repair Duration'].astype(np.float32)  # Ensure float32

# Categorical Columns
categorical_cols = ['Brand and model', 'Type', 'Shop']

# Numerical Columns (including derived time-based features and TF-IDF features)
numerical_cols = ['Year', 'Month', 'WeekOfYear', 'DayOfWeek', 'Quarter', 'IsWeekend'] + list(tfidf_df.columns)

# Define fixed categories for OneHotEncoder
unique_categories = {}
for col in categorical_cols:
    unique_categories[col] = df[col].unique().tolist()
    logging.info(f"Unique categories for '{col}': {unique_categories[col]}")

# Initialize OneHotEncoder with fixed categories
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', categories=[unique_categories[col] for col in categorical_cols]))
])

# Define preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing numerical values
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Define a full pipeline including preprocessing, feature counting, and casting to float32
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('feature_counter', FeatureCounter()),  # Logs feature count
    ('cast_to_float32', CastToFloat32())
])

# Split the data into training and testing sets based on time
split_ratio = 0.9
split_index = int(len(df) * split_ratio)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_duration_train, y_duration_test = y_duration.iloc[:split_index], y_duration.iloc[split_index:]
# Check for NaNs in y_duration_train and y_duration_test
if y_duration_train.isnull().any():
    num_nans_train = y_duration_train.isnull().sum()
    logging.error(f"'y_duration_train' contains {num_nans_train} NaN values.")
    # Decide how to handle them, e.g., drop
    y_duration_train = y_duration_train.dropna()
    X_train = X_train.loc[y_duration_train.index]
    logging.info("Dropped NaN values from 'y_duration_train' and aligned 'X_train'.")

if y_duration_test.isnull().any():
    num_nans_test = y_duration_test.isnull().sum()
    logging.error(f"'y_duration_test' contains {num_nans_test} NaN values.")
    # Handle them, e.g., drop
    y_duration_test = y_duration_test.dropna()
    X_test = X_test.loc[y_duration_test.index]
    logging.info("Dropped NaN values from 'y_duration_test' and aligned 'X_test'.")

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Fit the full pipeline on the entire training data to determine input_dim
full_pipeline.fit(X_train)
X_train_processed = full_pipeline.transform(X_train)
input_dim = X_train_processed.shape[1]
logging.info(f"Number of features after preprocessing: {input_dim}")

# Define multiple regression models and their hyperparameter grids
models = {
    'Linear Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    # 'Ridge Regression': {
    #     'model': Ridge(),
    #     'params': {
    #         'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
    #     }
    # },
    # 'Lasso Regression': {
    #     'model': Lasso(),
    #     'params': {
    #         'regressor__alpha': [0.1, 1.0, 10.0, 100.0]
    #     }
    # },
    # 'Elastic Net': {
    #     'model': ElasticNet(),
    #     'params': {
    #         'regressor__alpha': [0.1, 1.0, 10.0],
    #         'regressor__l1_ratio': [0.2, 0.5, 0.8]
    #     }
    # },
    # 'Decision Tree': {
    #     'model': DecisionTreeRegressor(random_state=42),
    #     'params': {
    #         'regressor__max_depth': [None, 5, 10, 20],
    #         'regressor__min_samples_split': [2, 5, 10]
    #     }
    # },
    # 'Random Forest': {
    #     'model': RandomForestRegressor(random_state=42),
    #     'params': {
    #         'regressor__n_estimators': [100, 200],
    #         'regressor__max_depth': [None, 10, 20],
    #         'regressor__min_samples_split': [2, 5]
    #     }
    # },
    # 'Gradient Boosting': {
    #     'model': GradientBoostingRegressor(random_state=42),
    #     'params': {
    #         'regressor__n_estimators': [100, 200],
    #         'regressor__learning_rate': [0.01, 0.1, 0.2],
    #         'regressor__max_depth': [3, 5, 10]
    #     }
    # },
    # 'AdaBoost': {
    #     'model': AdaBoostRegressor(random_state=42),
    #     'params': {
    #         'regressor__n_estimators': [50, 100, 200],
    #         'regressor__learning_rate': [0.01, 0.1, 1.0]
    #     }
    # },
    # 'Support Vector Regressor': {
    #     'model': SVR(),
    #     'params': {
    #         'regressor__C': [0.1, 1.0, 10.0],
    #         'regressor__epsilon': [0.1, 0.2, 0.5],
    #         'regressor__kernel': ['linear', 'rbf']
    #     }
    # },
    # 'K-Nearest Neighbors': {
    #     'model': KNeighborsRegressor(),
    #     'params': {
    #         'regressor__n_neighbors': [3, 5, 10],
    #         'regressor__weights': ['uniform', 'distance'],
    #         'regressor__p': [1, 2]
    #     }
    # },
    # # Adding PyTorch MLP via SafeNeuralNetRegressor and Skorch
    # 'PyTorch MLP': {
    #     'model': SafeNeuralNetRegressor(
    #         module=PyTorchMLPRegressor,
    #         module__input_dim=input_dim,  # Dynamic input_dim
    #         module__hidden_dims=[150, 75],
    #         module__activation=nn.ReLU,
    #         max_epochs=200,
    #         lr=0.001,
    #         optimizer=optim.Adam,
    #         criterion=nn.MSELoss(),
    #         device=device,  # 'cuda' or 'cpu'
    #         callbacks=[EarlyStopping(patience=10)],
    #         verbose=0
    #     ),
    #     'params': {
    #         'regressor__module__hidden_dims': [[150, 75], [200, 100], [100, 50]],
    #         'regressor__lr': [0.001, 0.01],
    #         'regressor__max_epochs': [200, 300],
    #     }
    # },
}

# Initialize a list to store model performances
model_performances = []

# Iterate over each model
for model_name, model_info in models.items():
    logging.info(f"Training and tuning {model_name}...")
    
    # Define the pipeline with preprocessing, feature counting, and casting
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_counter', FeatureCounter()),  # Logs feature count
        ('cast_to_float32', CastToFloat32()),
        ('regressor', model_info['model'])
    ])
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=model_info['params'],
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=0,
        error_score='raise'
    )
    
    try:
        # Fit the model
        grid_search.fit(X_train, y_duration_train)
    except Exception as e:
        logging.error(f"An error occurred while training {model_name}: {e}")
        continue  # Skip to the next model
    
    # Best parameters and CV score
    best_params = grid_search.best_params_
    best_cv_mae = -grid_search.best_score_
    
    logging.info(f"Best parameters for {model_name}: {best_params}")
    logging.info(f"Best CV MAE for {model_name}: {best_cv_mae:.2f} days")
    
    # Predict on the test set
    try:
        y_pred = grid_search.predict(X_test)
    except Exception as e:
        logging.error(f"An error occurred during prediction for {model_name}: {e}")
        continue
    # Check for NaNs in y_pred
    if np.isnan(y_pred).any():
        num_nans_pred = np.isnan(y_pred).sum()
        logging.error(f"'y_pred' contains {num_nans_pred} NaN values.")
        # Investigate the cause
        # For now, remove these predictions and corresponding true values
        valid_indices = ~np.isnan(y_pred)
        y_pred = y_pred[valid_indices]
        y_duration_test = y_duration_test.iloc[valid_indices]
        logging.info(f"Removed {num_nans_pred} NaN predictions and aligned 'y_duration_test'.")

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_duration_test, y_pred)
    median_ae = median_absolute_error(y_duration_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_duration_test, y_pred))
    r2 = r2_score(y_duration_test, y_pred)
    explained_var = explained_variance_score(y_duration_test, y_pred)
    
    # Handle MAPE: Avoid division by zero by excluding zero actual values
    non_zero_indices = y_duration_test != 0
    if np.sum(non_zero_indices) > 0:
        mape = mean_absolute_percentage_error(y_duration_test[non_zero_indices], y_pred[non_zero_indices]) * 100
    else:
        mape = np.nan  # Undefined if no non-zero actual values
    
    # Store the results
    model_performances.append({
        'Model': model_name,
        'Best Parameters': best_params,
        'Best CV MAE (days)': best_cv_mae,
        'Test MAE (days)': mae,
        'Test Median AE (days)': median_ae,
        'Test RMSE (days)': rmse,
        'Test R-squared': r2,
        'Test Explained Variance': explained_var,
        'Test MAPE (%)': mape
    })
    
    # Save the trained model
    if model_name != 'PyTorch MLP':
        model_filename = f"{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(grid_search.best_estimator_, model_filename)
        logging.info(f"Trained {model_name} saved as '{model_filename}'.")
    else:
        # For PyTorch MLP, save using torch
        try:
            model_module = grid_search.best_estimator_.named_steps['regressor'].module_
            torch.save(model_module.state_dict(), 'pytorch_mlp_model.pth')
            logging.info(f"Trained {model_name} saved as 'pytorch_mlp_model.pth'.")
        except AttributeError as e:
            logging.error(f"Failed to access 'module_': {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while saving the PyTorch MLP: {e}")

# Create a DataFrame from the performances
performance_df = pd.DataFrame(model_performances)

# Save the performance metrics to CSV
performance_df.to_csv('model_comparison_metrics.csv', index=False)
logging.info("Model comparison metrics saved to 'model_comparison_metrics.csv'.")

# Display the performance table
print("\n--- Model Comparison Metrics ---")
print(performance_df)
print("--------------------------------")

# Plotting the comparison of models based on MAE
plt.figure(figsize=(12, 8))
sns.barplot(x='Test MAE (days)', y='Model', data=performance_df.sort_values('Test MAE (days)'))
plt.title('Comparison of Models based on Test MAE')
plt.xlabel('Mean Absolute Error (days)')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('model_comparison_mae.png')
plt.close()
logging.info("Model comparison MAE plot saved as 'model_comparison_mae.png'.")

# Plotting the comparison of models based on R-squared
plt.figure(figsize=(12, 8))
sns.barplot(x='Test R-squared', y='Model', data=performance_df.sort_values('Test R-squared', ascending=False))
plt.title('Comparison of Models based on Test R-squared')
plt.xlabel('R-squared')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('model_comparison_r2.png')
plt.close()
logging.info("Model comparison R-squared plot saved as 'model_comparison_r2.png'.")

# Plotting the comparison of models based on RMSE
plt.figure(figsize=(12, 8))
sns.barplot(x='Test RMSE (days)', y='Model', data=performance_df.sort_values('Test RMSE (days)'))
plt.title('Comparison of Models based on Test RMSE')
plt.xlabel('Root Mean Squared Error (days)')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('model_comparison_rmse.png')
plt.close()
logging.info("Model comparison RMSE plot saved as 'model_comparison_rmse.png'.")

# Plotting the comparison of models based on MAPE
plt.figure(figsize=(12, 8))
sns.barplot(x='Test MAPE (%)', y='Model', data=performance_df.sort_values('Test MAPE (%)'))
plt.title('Comparison of Models based on Test MAPE')
plt.xlabel('Mean Absolute Percentage Error (%)')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('model_comparison_mape.png')
plt.close()
logging.info("Model comparison MAPE plot saved as 'model_comparison_mape.png'.")

# Save the TF-IDF vectorizer if not part of the pipeline
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
logging.info("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'.")
