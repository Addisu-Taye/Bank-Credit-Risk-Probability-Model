# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Identify high-risk customers using clustering and create a proxy target variable for model training.
# Key features:
# - Clusters customers into segments using KMeans based on RFM metrics
# - Identifies the high-risk cluster based on behavioral patterns
# - Merges high-risk labels back into original dataset
# - Prepares feature matrix and target vector for model training
# Task Number: 3

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


class FeatureEngineering:
    """
    A class to perform feature engineering on transaction data
    """

    def __init__(self):
        pass

    def load_data(self, file_path):
        """
        Load the raw data from Excel file
        """
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def extract_transaction_features(self, df):
        """
        Extract time-based features from transaction timestamps
        """
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Extract hour, day, month, year from timestamp
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        
        # Time since midnight in minutes
        df['minutes_since_midnight'] = (df['TransactionStartTime'].dt.hour * 60) + df['TransactionStartTime'].dt.minute
        
        # Weekend flag
        df['is_weekend'] = df['TransactionStartTime'].dt.dayofweek >= 5
        
        return df

    def create_aggregate_features(self, df):
        """
        Create aggregate features at customer level
        """
        # Calculate total transactions per customer
        transaction_count = df.groupby('CustomerId').size().reset_index(name='total_transactions')
        df = df.merge(transaction_count, on='CustomerId')
        
        # Calculate average transaction amount per customer
        avg_amount = df.groupby('CustomerId')['Amount'].mean().reset_index(name='avg_transaction_amount')
        df = df.merge(avg_amount, on='CustomerId')
        
        # Calculate standard deviation of transaction amounts
        std_amount = df.groupby('CustomerId')['Amount'].std().reset_index(name='std_transaction_amount')
        df = df.merge(std_amount, on='CustomerId')
        
        # Calculate total transaction value per customer
        total_value = df.groupby('CustomerId')['Value'].sum().reset_index(name='total_transaction_value')
        df = df.merge(total_value, on='CustomerId')
        
        # Calculate number of unique product categories per customer
        unique_categories = df.groupby('CustomerId')['ProductCategory'].nunique().reset_index(name='unique_product_categories')
        df = df.merge(unique_categories, on='CustomerId')
        
        return df

    def create_rfm_features(self, df, snapshot_date=None):
        """
        Create RFM (Recency, Frequency, Monetary) features
        """
        if snapshot_date is None:
            snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        rfm_df = df.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        })
        
        rfm_df.rename(columns={
            'TransactionStartTime': 'recency',
            'TransactionId': 'frequency',
            'Value': 'monetary'
        }, inplace=True)
        
        return rfm_df.reset_index()

    def preprocess_data(self, df):
        """
        Preprocess data with appropriate transformations
        """
        # Define numeric and categorical columns
        numeric_features = ['Amount', 'Value', 'total_transactions', 'avg_transaction_amount',
                           'std_transaction_amount', 'total_transaction_value', 'unique_product_categories']
        
        categorical_features = ['ChannelId', 'ProviderId', 'ProductCategory', 'PricingStrategy',
                               'CountryCode', 'transaction_hour', 'transaction_day', 'transaction_month']
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Apply transformations
        X = preprocessor.fit_transform(df)
        
        return X, preprocessor

    def process_data(self, raw_data_path):
        """
        Complete pipeline for processing data
        """
        # Load data
        df = self.load_data(raw_data_path)
        if df is None:
            return None, None
        
        # Extract transaction features
        df = self.extract_transaction_features(df)
        
        # Create aggregate features
        df = self.create_aggregate_features(df)
        
        # Create RFM features
        rfm_df = self.create_rfm_features(df)
        df = df.merge(rfm_df, on='CustomerId', how='left')
        
        # Return both full dataframe and RFM dataframe
        return df, rfm_df