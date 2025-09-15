#!/usr/bin/env python3
"""
A direct script entrypoint for manually triggering data collection from GitHub Actions.
This bypasses the Cloud Run service for easier debugging.
"""
import sys
from api.process_data import main

if __name__ == "__main__":
    print("--- Starting Manual Data Collection (Direct Script Run) ---")
    # We are forcing the run, which will use yfinance as the data source.
    result = main(force_run=True)

    print(f"--- Script Finished with Status: {result.get('status')} ---")
    print(f"Message: {result.get('message')}")

    if result.get('status') != 'success':
        print("❌ Script failed.")
        sys.exit(1)

    print("✅ Script completed successfully.")
    sys.exit(0)