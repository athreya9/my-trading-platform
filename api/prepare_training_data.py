#!/usr/bin/env python3
"""
This script prepares historical data for machine learning model training.

It reads the price data from the collected historical data, calculates all relevant technical
indicators to use as features (X), and defines a target variable (y)
that the model will learn to predict.
"""
import os
import sys
import pandas as pd
import numpy as np

from api.data_processing import calculate_indicators, apply_price_action_indicators
from api.config import ML_FEATURE_COLUMNS
from api.data_collector import DataCollector


# --- Configuration for the Target Variable ---
# How many periods in the future to look for a result.
PREDICTION_HORIZON = 5
# The minimum percentage move required to be considered a "win".
TARGET_RETURN_THRESHOLD = 0.002 # 0.2%

def create_target_variable(df):
    """
    Creates the target variable (y) for the machine learning model.

    The target is a binary classification:
    - 1 (BUY): If the price increases by TARGET_RETURN_THRESHOLD within the PREDICTION_HORIZON.
    - 0 (HOLD/SELL): Otherwise.
    """
    print(f"Creating target variable with a {PREDICTION_HORIZON}-period horizon and {TARGET_RETURN_THRESHOLD:.2%} return threshold...")

    def calculate_future_returns(group):
        """Helper function to apply on a per-instrument basis."""
        # Calculate the future return over the defined horizon
        future_returns = group['close'].shift(-PREDICTION_HORIZON) / group['close'] - 1
        # Create the binary target variable
        group['target'] = np.where(future_returns > TARGET_RETURN_THRESHOLD, 1, 0)
        return group

    # Apply the calculation for each instrument to avoid data leakage between symbols
    df = df.groupby('instrument', group_keys=False).apply(calculate_future_returns)

    # The last `PREDICTION_HORIZON` rows will have NaN targets, so we drop them.
    # This also drops NaNs created by the groupby operation.
    df.dropna(subset=['target'], inplace=True)
    df['target'] = df['target'].astype(int)

    print(f"Target variable created. Distribution:\n{df['target'].value_counts(normalize=True)}")
    return df


def main():
    """
    Main function to run the data preparation pipeline.
    """
    try:
        print("--- Starting ML Data Preparation ---")
        # 1. Read historical data
        collector = DataCollector()
        price_df = collector.fetch_historical_data("^NSEI", period="2y", interval="1d")
        if price_df is None or price_df.empty:
            print("Could not fetch historical data. Exiting.", file=sys.stderr)
            sys.exit(1)

        # Reset index to make sure 'Date' or 'Datetime' becomes a column
        price_df.reset_index(inplace=True)
        # Rename the date column to 'timestamp'
        if 'Date' in price_df.columns:
            price_df.rename(columns={'Date': 'timestamp'}, inplace=True)
        elif 'Datetime' in price_df.columns:
            price_df.rename(columns={'Datetime': 'timestamp'}, inplace=True)
        else:
            raise ValueError("No 'Date' or 'Datetime' column found in the historical data.")

        # 2. Calculate all indicators to use as features
        print("Calculating features (technical indicators)...")
        features_df = calculate_indicators(price_df)
        features_df = features_df.groupby('instrument', group_keys=False).apply(apply_price_action_indicators)

        # 3. Create the target variable
        labeled_df = create_target_variable(features_df)

        # 4. Select final features and clean the data
        final_columns = ML_FEATURE_COLUMNS + ['target']

        training_df = labeled_df[final_columns].copy()

        training_df[ML_FEATURE_COLUMNS] = training_df[ML_FEATURE_COLUMNS].fillna(0)

        training_df.dropna(inplace=True)

        # 5. Save the final dataset
        output_path = os.path.join(os.path.dirname(__file__), 'training_data.csv')
        training_df.to_csv(output_path, index=False)

        print("\n" + "="*50)
        print(f"✅ Success! Training data prepared and saved to:\n{output_path}")
        print(f"Total samples: {len(training_df)}")
        print("="*50)

    except Exception as e:
        print(f"\n❌ An error occurred during data preparation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()