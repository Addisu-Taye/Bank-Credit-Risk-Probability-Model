# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Clean raw transaction data (handle missing values, type conversion, filtering).
# Key features:
# - Load raw CSV file
# - Convert timestamps to datetime
# - Handle missing values
# - Filter relevant columns
# Task Number: 2

import pandas as pd
import os


def clean_data(input_path="data/raw/raw_data.csv", output_dir="data/processed"):
    """
    Clean raw transaction data and save cleaned version
    """
    print("Loading raw data...")
    df = pd.read_csv(input_path)

    # Filter out unnecessary columns
    cols_to_keep = [
        "TransactionId", "CustomerId", "Amount", "Value", "CountryCode",
        "ProviderId", "ProductCategory", "ChannelId", "TransactionStartTime",
        "PricingStrategy", "FraudResult"
    ]
    df = df[cols_to_keep]

    # Convert TransactionStartTime to datetime
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # Handle missing values
    missing_before = df.isna().sum()
    print("Missing values before cleaning:\n", missing_before)

    # Fill missing categorical values with 'Unknown'
    cat_cols = ["ProviderId", "ProductCategory", "ChannelId"]
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    # Drop rows where key numeric fields are missing
    df.dropna(subset=["Amount", "Value"], inplace=True)

    # Save cleaned data
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cleaned_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

    # Save missing value report
    missing_report_path = os.path.join("reports", "missing_values.csv")
    missing_after = df.isna().sum().to_frame(name="missing_count")
    missing_after.to_csv(missing_report_path)
    print(f"Missing value report saved to {missing_report_path}")

    return df


if __name__ == "__main__":
    df_cleaned = clean_data()