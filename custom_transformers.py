# custom_transformers.py

from sklearn.base import TransformerMixin, BaseEstimator
import logging
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np

class FeatureCounter(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        logging.info(f"Number of features after preprocessing: {X.shape[1]}")
        return X

class CastToFloat32(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.astype(np.float32)

class FeatureEngineer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Time-based features
        X['Year'] = X['Date'].dt.year
        X['Month'] = X['Date'].dt.month
        X['WeekOfYear'] = X['Date'].dt.isocalendar().week
        X['DayOfWeek'] = X['Date'].dt.dayofweek
        X['Quarter'] = X['Date'].dt.quarter
        X['IsWeekend'] = X['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Lag features
        X['lag_1'] = X['Repair Duration'].shift(1)
        X['lag_2'] = X['Repair Duration'].shift(2)
        X['rolling_mean_3'] = X['Repair Duration'].rolling(window=3).mean()
        
        # Handle NaNs
        X = X.bfill().ffill()
        
        return X

class TextPreprocessor(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def preprocess_text(text):
            text = str(text).lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(tokens)
        
        X['Processed_Repair_Description'] = X['Repair_Description'].apply(preprocess_text)
        return X
