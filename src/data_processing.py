# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Identify high-risk customers using clustering and create a proxy target variable for model training.
# Key features:
# - Clusters customers into segments using KMeans based on RFM metrics
# - Identifies the high-risk cluster based on behavioral patterns
# - Merges high-risk labels back into original dataset
# - Prepares feature matrix and target vector for model training
# - Saves key outputs to CSV files in a 'reports/' directory.
# - Generates and saves plots related to RFM distributions and customer clusters in a 'plots/' directory.
# Task Number: 4


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os # Import the os module for directory operations
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced plots


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

    def save_to_csv(self, df, filename, output_dir='reports/'):
        """
        Saves a DataFrame or Series to a CSV file.

        Args:
            df (pd.DataFrame or pd.Series): The DataFrame or Series to save.
            filename (str): The name of the CSV file (e.g., 'features.csv').
            output_dir (str): The directory to save the CSV file. Defaults to 'reports/'.
        """
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        filepath = os.path.join(output_dir, filename)
        if isinstance(df, pd.Series):
            df.name = df.name if df.name else 'value' # Ensure series has a name for CSV header
            df.to_csv(filepath, index=False, header=True)
        else:
            df.to_csv(filepath, index=False)
        print(f"Saved {filename} to {filepath}")

    def save_plot(self, fig, filename, output_dir='plots/'):
        """
        Saves a matplotlib figure to a file.

        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            filename (str): The name of the plot file (e.g., 'rfm_distributions.png').
            output_dir (str): The directory to save the plot. Defaults to 'plots/'.
        """
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches='tight') # bbox_inches='tight' prevents labels/titles from being cut off
        plt.close(fig) # Close the figure to free up memory
        print(f"Saved plot {filename} to {filepath}")

    def plot_rfm_distributions(self, rfm_df, output_dir='plots/'):
        """
        Plots the distributions of Recency, Frequency, and Monetary values.

        Args:
            rfm_df (pd.DataFrame): DataFrame containing 'recency', 'frequency', 'monetary'.
            output_dir (str): Directory to save the plots.
        """
        print("Generating RFM distribution plots...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5)) # Create a figure with 3 subplots

        sns.histplot(rfm_df['recency'], kde=True, ax=axes[0], color='skyblue')
        axes[0].set_title('Recency Distribution')
        axes[0].set_xlabel('Recency (Days)')
        axes[0].set_ylabel('Number of Customers')

        sns.histplot(rfm_df['frequency'], kde=True, ax=axes[1], color='lightcoral')
        axes[1].set_title('Frequency Distribution')
        axes[1].set_xlabel('Frequency (Transactions)')
        axes[1].set_ylabel('Number of Customers')

        sns.histplot(rfm_df['monetary'], kde=True, ax=axes[2], color='lightgreen')
        axes[2].set_title('Monetary Distribution')
        axes[2].set_xlabel('Monetary (Total Spend)')
        axes[2].set_ylabel('Number of Customers')

        plt.tight_layout() # Adjust subplot parameters for a tight layout
        self.save_plot(fig, 'rfm_distributions.png', output_dir)
        print("RFM distribution plots generated.")

    def plot_cluster_scatter(self, rfm_df, high_risk_cluster, output_dir='plots/'):
        """
        Plots scatter plots of RFM features, colored by cluster,
        highlighting the high-risk cluster.

        Args:
            rfm_df (pd.DataFrame): DataFrame with 'recency', 'frequency', 'monetary', 'cluster', 'is_high_risk'.
            high_risk_cluster (int): The label of the identified high-risk cluster.
            output_dir (str): Directory to save the plots.
        """
        print("Generating cluster scatter plots...")

        # Plot 1: Recency vs Frequency
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='recency', y='frequency', hue='cluster', data=rfm_df,
                        palette='viridis', ax=ax1, s=50, alpha=0.7)
        # Highlight high-risk customers with a distinct marker and color
        sns.scatterplot(x='recency', y='frequency', data=rfm_df[rfm_df['is_high_risk'] == 1],
                        color='red', marker='X', s=200, label='High-Risk', ax=ax1, zorder=5) # zorder to ensure it's on top
        ax1.set_title('Customer Clusters: Recency vs Frequency')
        ax1.set_xlabel('Recency (Days)')
        ax1.set_ylabel('Frequency (Transactions)')
        ax1.legend(title='Cluster')
        self.save_plot(fig1, 'cluster_recency_frequency.png', output_dir)

        # Plot 2: Recency vs Monetary
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='recency', y='monetary', hue='cluster', data=rfm_df,
                        palette='viridis', ax=ax2, s=50, alpha=0.7)
        sns.scatterplot(x='recency', y='monetary', data=rfm_df[rfm_df['is_high_risk'] == 1],
                        color='red', marker='X', s=200, label='High-Risk', ax=ax2, zorder=5)
        ax2.set_title('Customer Clusters: Recency vs Monetary')
        ax2.set_xlabel('Recency (Days)')
        ax2.set_ylabel('Monetary (Total Spend)')
        ax2.legend(title='Cluster')
        self.save_plot(fig2, 'cluster_recency_monetary.png', output_dir)

        # Plot 3: Frequency vs Monetary
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x='frequency', y='monetary', hue='cluster', data=rfm_df,
                        palette='viridis', ax=ax3, s=50, alpha=0.7)
        sns.scatterplot(x='frequency', y='monetary', data=rfm_df[rfm_df['is_high_risk'] == 1],
                        color='red', marker='X', s=200, label='High-Risk', ax=ax3, zorder=5)
        ax3.set_title('Customer Clusters: Frequency vs Monetary')
        ax3.set_xlabel('Frequency (Transactions)')
        ax3.set_ylabel('Monetary (Total Spend)')
        ax3.legend(title='Cluster')
        self.save_plot(fig3, 'cluster_frequency_monetary.png', output_dir)

        print("Cluster scatter plots generated.")

    def process_pipeline(self, raw_data_df):
        """
        Full pipeline for data processing, including RFM calculation, clustering,
        and proxy target variable creation, aligned with Task 4.
        Also saves key outputs to CSV files and plots.

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
        try:
            fe = FeatureEngineering()
        except NameError:
            print("Error: FeatureEngineering class not found. Please ensure it's defined and imported.")
            raise

        # Process data through feature engineering to get the main DataFrame and RFM DataFrame
        df, rfm_df = fe.process_data(raw_data_df.copy())

        # Plot RFM distributions before clustering
        # Pass a copy of rfm_df to ensure the plotting function does not modify the original
        self.plot_rfm_distributions(rfm_df.copy())

        # Identify high risk customers using K-Means clustering on RFM features
        rfm_df, cluster_analysis, high_risk_cluster, kmeans_model, scaler = self.identify_high_risk_customers(rfm_df)

        if rfm_df is None or rfm_df.empty or 'is_high_risk' not in rfm_df.columns:
            print("Error: Failed to identify high-risk customers or 'is_high_risk' column is missing after clustering.")
            return None, None, None, None, None, None

        # Plot customer clusters after high-risk identification
        # Pass a copy of rfm_df to ensure the plotting function does not modify the original
        self.plot_cluster_scatter(rfm_df.copy(), high_risk_cluster)

        # Merge the 'is_high_risk' indicator back into the original dataset
        processed_df = self.merge_with_original_data(df, rfm_df)

        # Prepare the final feature matrix (X) and target vector (y) for model training
        X, y = self.prepare_training_data(processed_df)

        # Save outputs to CSV files
        self.save_to_csv(X, 'features.csv')
        self.save_to_csv(y, 'target_variable.csv')
        self.save_to_csv(cluster_analysis, 'cluster_analysis.csv')
        # Save the full processed_df which includes CustomerId, RFM, and is_high_risk
        self.save_to_csv(processed_df, 'processed_customer_data.csv')

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
