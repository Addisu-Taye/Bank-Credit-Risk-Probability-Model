from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime

# Import local modules
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from predict import RiskPredictor
from feature_engineering import FeatureEngineering
from data_processing import DataProcessing

# Initialize app
app = FastAPI(title="Credit Risk API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = RiskPredictor()

# Define Pydantic models
class Transaction(BaseModel):
    Amount: float
    Value: float
    CountryCode: int
    ProviderId: int
    PricingStrategy: int
    ProductCategory: int
    ChannelId: int
    TransactionStartTime: str

class CustomerRiskRequest(BaseModel):
    transactions: List[Transaction]

class RiskPredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool

@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Risk Prediction API"}

@app.post("/predict", response_model=RiskPredictionResponse)
def predict_credit_risk(customer_data: CustomerRiskRequest):
    """
    Predict credit risk for a customer based on their transaction history
    
    Input should be a JSON object with an array of transactions.
    Each transaction should contain:
    - Amount: Transaction amount (positive for debits, negative for credits)
    - Value: Absolute value of transaction amount
    - CountryCode: Numerical country code
    - ProviderId: ID of the service provider
    - PricingStrategy: Type of pricing strategy used
    - ProductCategory: Category of product/service purchased
    - ChannelId: Channel used for transaction
    - TransactionStartTime: Timestamp of transaction (ISO format)
    """
    try:
        # Convert transaction data to DataFrame
        df = pd.DataFrame([t.dict() for t in customer_data.transactions])
        
        # Convert TransactionStartTime to datetime
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Apply feature engineering
        fe = FeatureEngineering()
        df = fe.extract_transaction_features(df)
        
        # Create RFM features
        dp = DataProcessing()
        rfm_df = dp.create_rfm_features(df)
        
        # For prediction, we need to take the latest state of the customer
        # Get the most recent transaction details
        latest_transaction = df.loc[df['TransactionStartTime'].idxmax()]
        
        # Fill NaN values with 0 for std calculation
        df['Amount'] = df['Amount'].fillna(0)
        
        # Create a single row with combined features
        input_data = pd.DataFrame({
            'Amount': [latest_transaction['Amount']],
            'Value': [latest_transaction['Value']],
            'CountryCode': [latest_transaction['CountryCode']],
            'ProviderId': [latest_transaction['ProviderId']],
            'PricingStrategy': [latest_transaction['PricingStrategy']],
            'ProductCategory': [latest_transaction['ProductCategory']],
            'ChannelId': [latest_transaction['ChannelId']],
            'transaction_hour': [latest_transaction['transaction_hour']],
            'transaction_day': [latest_transaction['transaction_day']],
            'transaction_month': [latest_transaction['transaction_month']],
            'total_transactions': [len(df)],
            'avg_transaction_amount': [df['Amount'].mean()],
            'std_transaction_amount': [df['Amount'].std() if len(df) > 1 else 0],
            'total_transaction_value': [df['Value'].sum()],
            'unique_product_categories': [df['ProductCategory'].nunique()],
            'recency': [rfm_df['recency'].values[0] if len(rfm_df) > 0 else 0],
            'frequency': [rfm_df['frequency'].values[0] if len(rfm_df) > 0 else 0],
            'monetary': [rfm_df['monetary'].values[0] if len(rfm_df) > 0 else 0]
        }).fillna(0)  # Fill any remaining NaN values
        
        # Make prediction
        risk_probability = predictor.predict_risk_probability(input_data)[0]
        is_high_risk = risk_probability > 0.5  # Assuming 0.5 threshold
        
        return {
            "risk_probability": round(float(risk_probability), 4),
            "is_high_risk": bool(is_high_risk)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))