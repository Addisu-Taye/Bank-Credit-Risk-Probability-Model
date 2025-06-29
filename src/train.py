# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Train, evaluate, and track multiple machine learning models to predict credit risk probability.
# Key features:
# - Splits data into train/test sets
# - Trains Logistic Regression, Random Forest, and Gradient Boosting models
# - Performs hyperparameter tuning via GridSearchCV
# - Evaluates models using accuracy, precision, recall, F1, and ROC-AUC
# - Logs results and registers best model in MLflow
# Task Number: 5

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import pytest
import os
import pickle


class ModelTraining:
    """
    A class to handle model training and evaluation
    """

    def __init__(self):
        # Set up MLflow tracking
        mlflow.set_tracking_uri("http://localhost:5000")
        os.environ["MLFLOW_REGISTRY_URI"] = "./mlruns"
        mlflow.set_experiment("Credit Risk Modeling")

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def train_logistic_regression(self, X_train, y_train):
        """
        Train a logistic regression model
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        param_grid = {
            'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'clf__penalty': ['l1', 'l2'],
            'clf__solver': ['liblinear']
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def train_random_forest(self, X_train, y_train):
        """
        Train a random forest classifier
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        
        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 5, 10],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def train_gradient_boosting(self, X_train, y_train):
        """
        Train a gradient boosting classifier
        """
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(random_state=42))
        ])
        
        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__learning_rate': [0.01, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7],
            'clf__min_samples_split': [2, 5],
            'clf__min_samples_leaf': [1, 2]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate model performance
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        return metrics

    def log_model_mlflow(self, model, params, metrics, model_name):
        """
        Log model and metrics to MLflow
        """
        with mlflow.start_run():
            mlflow.set_tag("model", model_name)
            
            # Log parameters
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            mv = mlflow.register_model(model_uri, model_name)
            
        return mv.version

    def train_best_model(self, X_train, y_train, model_type="random_forest"):
        """
        Train the best model based on specified type
        """
        if model_type == "logistic_regression":
            return self.train_logistic_regression(X_train, y_train)
        elif model_type == "random_forest":
            return self.train_random_forest(X_train, y_train)
        elif model_type == "gradient_boosting":
            return self.train_gradient_boosting(X_train, y_train)
        else:
            raise ValueError("Unsupported model type")

    def save_preprocessor(self, preprocessor, filename="preprocessor.pkl"):
        """
        Save the preprocessor to disk
        """
        with open(filename, 'wb') as f:
            pickle.dump(preprocessor, f)
        return filename

    def training_pipeline(self, X, y, model_type="random_forest"):
        """
        Full training pipeline
        """
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Train best model
        model, params, best_score = self.train_best_model(X_train, y_train, model_type)
        
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Log model in MLflow
        version = self.log_model_mlflow(model, params, metrics, model_type)
        
        return model, metrics, version