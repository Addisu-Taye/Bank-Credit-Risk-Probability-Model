# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Load trained model and preprocessor to make predictions on new customer data.
# Key features:
# - Loads model from MLflow registry
# - Loads preprocessing pipeline
# - Applies preprocessing to input data
# - Predicts risk probability and classifies customers as high/low risk
# Task Number: 6
import mlflow
import mlflow.sklearn
import pickle
import os


class RiskPredictor:
    """
    A class to handle risk probability predictions
    """
    
    def __init__(self, model_name="random_forest"):
        # Set up MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5000")
        os.environ["MLFLOW_REGISTRY_URI"] = "./mlruns"
        
        # Load preprocessor
        with open("preprocessor.pkl", 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load model
        self.model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data before making predictions
        """
        # Apply same preprocessing as training
        processed_data = self.preprocessor.transform(input_data)
        return processed_data
    
    def predict_risk_probability(self, input_data):
        """
        Predict risk probability for new customer data
        """
        processed_data = self.preprocess_input(input_data)
        probabilities = self.model.predict_proba(processed_data)[:, 1]
        return probabilities
    
    def batch_prediction(self, batch_data):
        """
        Make predictions on a batch of data
        """
        return self.predict_risk_probability(batch_data)