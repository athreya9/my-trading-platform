#!/usr/bin/env python3
"""
This script prepares historical data for machine learning model training.

It reads the price data from Google Sheets, calculates all relevant technical
indicators to use as features (X), and defines a target variable (y)
that the model will learn to predict.
"""
import os
import sys
import pandas as pd
import numpy as np

# Add the parent directory to the path to allow imports from 'api'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import read_price_data, connect_to_google_sheets
from api.process_data import calculate_indicators, apply_price_action_indicators


# --- Configuration for the Target Variable ---
# How many periods in the future to look for a result.
PREDICTION_HORIZON = 5
# The minimum percentage move required to be considered a "win".
TARGET_RETURN_THRESHOLD = 0.005 # 0.5%


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
        # 1. Read historical data from Google Sheets
        spreadsheet = connect_to_google_sheets()
        # Read data for ALL instruments to create a richer training set
        price_df = read_price_data(spreadsheet, target_instrument=None)

        # 2. Calculate all indicators to use as features
        # We reuse the robust functions from the main processing script.
        print("Calculating features (technical indicators)...")
        features_df = calculate_indicators(price_df)
        features_df = features_df.groupby('instrument', group_keys=False).apply(apply_price_action_indicators)

        # 3. Create the target variable
        labeled_df = create_target_variable(features_df)

        # 4. Select final features and clean the data
        # We drop columns that are not useful for prediction (like future data or raw prices).
        # This is a starting point; feature selection is a key part of ML.
        feature_columns = [
            'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9', 'ATRr_14',
            'volume_avg_20', 'realized_vol', 'vwap', 'bos', 'choch',
            'last_bull_ob_top', 'last_bull_ob_bottom'
        ]
        final_columns = feature_columns + ['target']

        # Drop rows with any NaN values in the selected feature columns
        training_df = labeled_df[final_columns].dropna()

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