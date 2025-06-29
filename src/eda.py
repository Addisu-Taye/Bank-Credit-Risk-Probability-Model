# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Perform Exploratory Data Analysis (EDA) and generate plots.
# Key features:
# - Generate histograms and boxplots for numerical variables
# - Analyze categorical variable distributions
# - Compute correlations
# - Save plots and summary stats
# Task Number: 2

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def run_eda(input_path="data/processed/cleaned_data.csv", plot_dir="plots", report_dir="reports"):
    """
    Run EDA on cleaned data and save results
    """
    print("Loading cleaned data...")
    df = pd.read_csv(input_path)

    # Create directories
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Summary Statistics
    summary = df.describe(include='all').T
    summary_path = os.path.join(report_dir, "eda_summary.csv")
    summary.to_csv(summary_path)
    print(f"Summary stats saved to {summary_path}")

    # Correlation Matrix
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(plot_dir, "correlation_matrix.png"))
    plt.close()

    # Histograms for numerical features
    num_cols = ['Amount', 'Value']
    for col in num_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(plot_dir, f"histogram_{col}.png"))
        plt.close()

    # Boxplots for outliers
    for col in num_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(os.path.join(plot_dir, f"boxplot_{col}.png"))
        plt.close()

    # Categorical distribution plots
    cat_cols = ["ProductCategory", "ProviderId", "ChannelId"]
    for col in cat_cols:
        top_categories = df[col].value_counts().head(10).index
        subset = df[df[col].isin(top_categories)]
        plt.figure(figsize=(10, 6))
        sns.countplot(data=subset, y=col, order=top_categories)
        plt.title(f"Top 10 Categories in {col}")
        plt.xlabel("Count")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"barplot_{col}.png"))
        plt.close()

    print("EDA completed successfully.")


if __name__ == "__main__":
    run_eda()