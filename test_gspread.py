import gspread
import json
import os

creds_file_path = "/Users/datta/Desktop/My trading platform/google_sheets_credentials.json"

try:
    client = gspread.service_account(filename=creds_file_path)
    print("Successfully connected to gspread.")
except Exception as e:
    print(f"Error connecting to gspread: {e}")
