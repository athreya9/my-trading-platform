#!/usr/bin/env python3
"""
A dedicated script to initialize or repair the Google Sheet structure.
This ensures all required tabs and headers are present.
"""
import sys
import logging
from dotenv import load_dotenv

# Load .env for local runs
load_dotenv()

# Add the parent directory to the path to allow imports from 'api'
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the shared utility function
from api.sheet_utils import connect_to_google_sheets, enhance_sheet_structure

# --- Configuration ---
SHEET_NAME = "Algo Trading Dashboard"

# --- Logging Configuration ---
formatter = logging.Formatter('%(asctime)s UTC - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def main():
    """Connects to the sheet and runs the structure enhancement."""
    try:
        logger.info(f"--- Starting Google Sheet Setup for '{SHEET_NAME}' ---")
        spreadsheet = connect_to_google_sheets(SHEET_NAME)
        enhance_sheet_structure(spreadsheet)
        logger.info("✅ --- Google Sheet setup completed successfully! ---")
    except Exception as e:
        logger.error(f"❌ A critical error occurred during sheet setup: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()