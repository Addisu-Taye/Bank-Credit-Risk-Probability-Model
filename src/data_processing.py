# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Identify high-risk customers using clustering and create a proxy target variable for model training.
# Key features:
# - Clusters customers into segments using KMeans based on RFM metrics
# - Identifies the high-risk cluster based on behavioral patterns
# - Merges high-risk labels back into original dataset
# - Prepares feature matrix and target vector for model training
# Task Number: 4


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class DataProcessing:
    """
    A class to handle data processing tasks including proxy target creation
    for credit risk assessment.
    """

    def __init__(self):
        pass

    def identify_high_risk_customers(self, rfm_df, n_clusters=3, random_state=42):
        """
        Identifies high-risk customers using K-means clustering on RFM features.

        High-risk customers are defined as the least engaged segment, typically
        characterized by low frequency, low monetary value, and higher recency.

        Args:
            rfm_df (pd.DataFrame): DataFrame containing 'recency', 'frequency',
                                   'monetary' for each 'CustomerId'.
            n_clusters (int): The number of clusters for KMeans. Defaults to 3.
            random_state (int): Seed for reproducibility of KMeans clustering.

        Returns:
            tuple:
                - rfm_df (pd.DataFrame): The RFM DataFrame with 'cluster' labels
                                         and 'is_high_risk' binary indicator.
                - cluster_analysis (pd.DataFrame): Summary statistics for each cluster.
                - high_risk_cluster (int): The label of the identified high-risk cluster.
                - kmeans_model (sklearn.cluster.KMeans): The trained KMeans model.
                - scaler (sklearn.preprocessing.StandardScaler): The scaler used for RFM features.
        """
        if rfm_df.empty:
            print("Warning: RFM DataFrame is empty. Cannot identify high-risk customers.")
            return rfm_df, None, None, None, None

        # Check if RFM features exist
        required_rfm_cols = ['recency', 'frequency', 'monetary']
        if not all(col in rfm_df.columns for col in required_rfm_cols):
            raise ValueError(f"RFM DataFrame must contain '{required_rfm_cols}' columns.")

        # Handle potential infinite values in RFM, which can arise from 1/0 operations
        # Replace inf with NaN and then drop rows where RFM values are NaN
        rfm_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        rfm_df.dropna(subset=required_rfm_cols, inplace=True)

        if rfm_df.empty:
            print("Warning: RFM DataFrame became empty after handling missing/infinite values. Cannot identify high-risk customers.")
            return rfm_df, None, None, None, None

        # Scale the RFM features to ensure all features contribute equally to clustering
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_df[required_rfm_cols])

        # Apply K-means clustering
        # n_init='auto' or a number (e.g., 10) is recommended for KMeans stability
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        cluster_labels = kmeans.fit_predict(rfm_scaled)

        # Add cluster labels to RFM DataFrame
        rfm_df['cluster'] = cluster_labels

        # Analyze clusters to understand their characteristics
        cluster_analysis = rfm_df.groupby('cluster')[required_rfm_cols].mean()
        cluster_analysis['count'] = rfm_df['cluster'].value_counts().sort_index()

        # Identify which cluster represents high-risk customers
        # High risk customers typically have low frequency, low monetary value, and higher recency.
        # We assign scores to clusters: higher recency contributes positively, lower frequency
        # and monetary value (represented by 1/value) contribute positively to the score.
        cluster_scores = []
        for cluster in range(n_clusters):
            # Ensure frequency and monetary are not zero to avoid division by zero
            freq = cluster_analysis.loc[cluster, 'frequency']
            mon = cluster_analysis.loc[cluster, 'monetary']

            # Assign a high score if frequency or monetary is zero to mark it as potentially high risk
            if freq == 0 or mon == 0:
                score = np.inf
            else:
                score = (
                    cluster_analysis.loc[cluster, 'recency'] * 0.3 +        # Recency: higher is worse
                    (1 / freq) * 0.4 +                                    # Frequency: lower is worse (higher 1/freq)
                    (1 / mon) * 0.3                                       # Monetary: lower is worse (higher 1/mon)
                )
            cluster_scores.append(score)

        # The cluster with the highest score is identified as the high-risk cluster
        high_risk_cluster = np.argmax(cluster_scores)

        # Create binary high-risk indicator (proxy target variable)
        rfm_df['is_high_risk'] = (rfm_df['cluster'] == high_risk_cluster).astype(int)

        print(f"High-risk cluster identified: {high_risk_cluster}")

        return rfm_df, cluster_analysis, high_risk_cluster, kmeans, scaler

    def merge_with_original_data(self, original_df, rfm_df):
        """
        Merges the 'is_high_risk' indicator back into the original dataset.

        Args:
            original_df (pd.DataFrame): The original DataFrame containing customer data.
            rfm_df (pd.DataFrame): The RFM DataFrame with 'CustomerId' and 'is_high_risk' columns.

        Returns:
            pd.DataFrame: The original DataFrame enriched with the 'is_high_risk' column.
        """
        # Ensure 'CustomerId' is present in both dataframes for a successful merge
        if 'CustomerId' not in original_df.columns:
            raise ValueError("Original DataFrame must contain 'CustomerId' column for merging.")
        if 'CustomerId' not in rfm_df.columns:
            raise ValueError("RFM DataFrame must contain 'CustomerId' column for merging.")

        processed_df = original_df.merge(
            rfm_df[['CustomerId', 'is_high_risk']],
            on='CustomerId',
            how='left' # Use a left merge to keep all original customers
        )

        # Fill any missing values in 'is_high_risk' with 0. This handles cases where
        # a CustomerId in original_df might not have been present in rfm_df (e.g.,
        # no transactions, hence no RFM calculated). Such customers are assumed not high-risk.
        initial_missing_risk = processed_df['is_high_risk'].isnull().sum()
        processed_df['is_high_risk'] = processed_df['is_high_risk'].fillna(0).astype(int)
        if initial_missing_risk > 0:
            print(f"Filled {initial_missing_risk} missing 'is_high_risk' values with 0 after merge.")

        return processed_df

    def prepare_training_data(self, processed_df):
        """
        Prepares training data by selecting relevant features and the target variable.

        Args:
            processed_df (pd.DataFrame): The DataFrame after merging the 'is_high_risk' column.

        Returns:
            tuple:
                - X (pd.DataFrame): Feature matrix for model training.
                - y (pd.Series): Target vector ('is_high_risk').
        """
        # Define features that make sense for credit risk prediction, including RFM metrics
        selected_features = [
            'Amount', 'Value', 'CountryCode', 'ProviderId',
            'PricingStrategy', 'transaction_hour', 'transaction_day',
            'transaction_month', 'ProductCategory', 'ChannelId',
            'total_transactions', 'avg_transaction_amount',
            'std_transaction_amount', 'total_transaction_value',
            'unique_product_categories', 'recency', 'frequency', 'monetary'
        ]

        # Filter out any selected features that are not present in the DataFrame
        # This makes the function more robust to variations in input data
        available_features = [col for col in selected_features if col in processed_df.columns]
        missing_features = [col for col in selected_features if col not in processed_df.columns]
        if missing_features:
            print(f"Warning: The following selected features are missing from the DataFrame and will be skipped: {missing_features}")

        if 'is_high_risk' not in processed_df.columns:
            raise ValueError("Target variable 'is_high_risk' not found in processed DataFrame. Ensure it was merged.")

        X = processed_df[available_features]
        y = processed_df['is_high_risk']

        print(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
        return X, y

    def process_pipeline(self, raw_data_df):
        """
        Full pipeline for data processing, including RFM calculation, clustering,
        and proxy target variable creation, aligned with Task 4.

        Args:
            raw_data_df (pd.DataFrame): The raw input DataFrame containing transaction data.

        Returns:
            tuple:
                - X (pd.DataFrame): Feature matrix for model training.
                - y (pd.Series): Target vector.
                - cluster_analysis (pd.DataFrame): Analysis of customer clusters.
                - high_risk_cluster (int): The identified high-risk cluster label.
                - kmeans_model (sklearn.cluster.KMeans): The trained KMeans model.
                - scaler (sklearn.preprocessing.StandardScaler): The scaler used for RFM features.
        """
        print("Starting full data processing pipeline for Task 4: Proxy Target Variable Engineering.")

        # Initialize feature engineering (assuming FeatureEngineering class is defined elsewhere)
        # This part relies on an external FeatureEngineering class to prepare 'df' and 'rfm_df'.
        try:
            fe = FeatureEngineering()
        except NameError:
            print("Error: FeatureEngineering class not found. Please ensure it's defined and imported.")
            raise

        # Process data through feature engineering to get the main DataFrame and RFM DataFrame
        # The 'raw_data_df' is passed directly as cleaning is not part of this task.
        df, rfm_df = fe.process_data(raw_data_df.copy()) # Use .copy() to avoid modifying original df

        # Identify high risk customers using K-Means clustering on RFM features
        rfm_df, cluster_analysis, high_risk_cluster, kmeans_model, scaler = self.identify_high_risk_customers(rfm_df)

        if rfm_df is None or rfm_df.empty or 'is_high_risk' not in rfm_df.columns:
            print("Error: Failed to identify high-risk customers or 'is_high_risk' column is missing after clustering.")
            return None, None, None, None, None, None

        # Merge the 'is_high_risk' indicator back into the original dataset
        processed_df = self.merge_with_original_data(df, rfm_df)

        # Prepare the final feature matrix (X) and target vector (y) for model training
        X, y = self.prepare_training_data(processed_df)

        print("Full data processing pipeline for Task 4 complete.")
        return X, y, cluster_analysis, high_risk_cluster, kmeans_model, scaler

# Example Usage (assuming a FeatureEngineering class and dummy data for demonstration)
if __name__ == "__main__":
    # Dummy FeatureEngineering class to make the example runnable
    # In a real scenario, this would be a separate, more complex class
    class FeatureEngineering:
        def process_data(self, df):
            # Simulate some basic feature creation and RFM calculation
            # Ensure 'TransactionDate' is datetime
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

            # Example: derive time-based features (needed for prepare_training_data)
            df['transaction_hour'] = df['TransactionDate'].dt.hour
            df['transaction_day'] = df['TransactionDate'].dt.day
            df['transaction_month'] = df['TransactionDate'].dt.month

            # Calculate aggregate features per customer
            customer_agg = df.groupby('CustomerId').agg(
                total_transactions=('TransactionId', 'count'),
                total_transaction_value=('Amount', 'sum'),
                avg_transaction_amount=('Amount', 'mean'),
                std_transaction_amount=('Amount', 'std'),
                unique_product_categories=('ProductCategory', lambda x: x.nunique())
            ).reset_index()

            # Calculate RFM features
            snapshot_date = df['TransactionDate'].max() + pd.Timedelta(days=1)
            rfm = df.groupby('CustomerId').agg(
                recency=('TransactionDate', lambda date: (snapshot_date - date.max()).days),
                frequency=('TransactionId', 'count'),
                monetary=('Amount', 'sum')
            ).reset_index()

            # Merge aggregate features back into the main DataFrame 'df' for 'prepare_training_data'
            df = df.merge(customer_agg, on='CustomerId', how='left')

            return df, rfm

    # Create dummy raw data for demonstration
    data = {
        'CustomerId': [1, 1, 2, 2, 3, 4, 1, 5, 5, 6, 7, 7, 8],
        'TransactionId': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113],
        'TransactionDate': pd.to_datetime([
            '2024-01-01', '2024-01-15', '2024-02-01', '2024-02-10', '2024-03-01',
            '2024-03-10', '2024-01-05', '2024-04-01', '2024-04-05', '2024-05-01',
            '2024-05-05', '2024-05-06', '2024-06-01'
        ]),
        'Amount': [100, 50, 200, 75, 300, 10, 60, 150, 25, 400, 80, 120, 5],
        'Value': [10, 5, 20, 7, 30, 1, 6, 15, 2, 40, 8, 12, 0.5],
        'CountryCode': ['US', 'US', 'CA', 'CA', 'MX', 'US', 'US', 'DE', 'DE', 'FR', 'FR', 'FR', 'GB'],
        'ProviderId': ['A', 'B', 'A', 'C', 'B', 'A', 'A', 'C', 'A', 'B', 'C', 'A', 'B'],
        'PricingStrategy': [1, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'ProductCategory': ['Electronics', 'Books', 'Electronics', 'Home', 'Books',
                            'Groceries', 'Electronics', 'Books', 'Groceries', 'Home',
                            'Electronics', 'Electronics', 'Books'],
        'ChannelId': ['Online', 'Store', 'Online', 'Store', 'Online', 'Store',
                      'Online', 'Store', 'Online', 'Store', 'Online', 'Store', 'Online']
    }
    raw_df = pd.DataFrame(data)

    print("Original raw data head:")
    print(raw_df.head())

    # Instantiate and run the pipeline
    processor = DataProcessing()
    try:
        X_train, y_train, cluster_summary, high_risk_c, kmeans_m, scaler_m = processor.process_pipeline(raw_df.copy())

        print("\n--- Processing Results ---")
        if X_train is not None and y_train is not None:
            print("\nShape of X_train:", X_train.shape)
            print("Shape of y_train:", y_train.shape)
            print("\nFirst 5 rows of X_train:")
            print(X_train.head())
            print("\nFirst 5 rows of y_train:")
            print(y_train.head())
            print("\nCluster Analysis:\n", cluster_summary)
            print(f"\nIdentified High-Risk Cluster Label: {high_risk_c}")
        else:
            print("Pipeline did not return valid training data.")

    except Exception as e:
        print(f"\nAn error occurred during pipeline execution: {e}")

    print("\nDemonstrating 'is_high_risk' in the merged RFM DataFrame:")
    # To show the 'is_high_risk' column directly from rfm_df, we need to rerun a part of the pipeline
    # just for demonstration purposes outside the main pipeline return.
    temp_fe = FeatureEngineering()
    _, temp_rfm_df = temp_fe.process_data(raw_df.copy())
    temp_rfm_df, _, _, _, _ = processor.identify_high_risk_customers(temp_rfm_df)
    print(temp_rfm_df[['CustomerId', 'recency', 'frequency', 'monetary', 'cluster', 'is_high_risk']])
