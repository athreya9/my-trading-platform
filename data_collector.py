# data_collector.py

# The libraries we need to use
import gspread
import yfinance as yf
import pandas as pd

# --- Configuration ---
# It's good practice to keep configurable variables at the top.
SHEET_NAME = "Algo Trading Dashboard"
WORKSHEET_NAME = "Price Data"
SERVICE_ACCOUNT_FILE = 'service_account.json'
# Note: 'yfinance' uses a different format for Indian stocks.
SYMBOLS = ['RELIANCE.NS', '^NSEI'] 
# Define headers once to ensure consistency.
SHEET_HEADERS = ['timestamp', 'instrument', 'open', 'high', 'low', 'close', 'volume']


def connect_to_google_sheets():
    """Connects to Google Sheets using the service account credentials."""
    try:
        # The 'oauth2client' library is deprecated. 
        # 'gspread.service_account' is the modern, recommended way to authenticate.
        client = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)

        # Open the Google Sheet and the specific worksheet
        sheet = client.open(SHEET_NAME).worksheet(WORKSHEET_NAME)
        print("Successfully connected to Google Sheet.")
        return sheet
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"Error: Spreadsheet '{SHEET_NAME}' not found. Please check the name.")
        exit()
    except gspread.exceptions.WorksheetNotFound:
        print(f"Error: Worksheet '{WORKSHEET_NAME}' not found in '{SHEET_NAME}'.")
        exit()
    except Exception as e:
        print(f"An error occurred while connecting to Google Sheets: {e}")
        # Exit the script if we can't connect, as there's nothing else to do.
        exit()


def fetch_historical_data(ticker, period='5d', interval='15m'):
    """Fetches historical data from Yahoo Finance for a given ticker."""
    print(f"Fetching data for {ticker}...")
    try:
        # Download the data using yfinance
        # We add auto_adjust=True to silence the FutureWarning and use the modern default.
        stock_data = yf.download(
            tickers=ticker, period=period, interval=interval, auto_adjust=True
        )

        if stock_data.empty:
            print(f"No data downloaded for {ticker}. It might be an invalid symbol or delisted.")
            return pd.DataFrame()

        stock_data.reset_index(inplace=True)

        # Build a new DataFrame from scratch to ensure a consistent structure.
        # This is more robust than renaming and slicing, and prevents column mismatches.
        clean_df = pd.DataFrame()
        clean_df['timestamp'] = stock_data['Datetime']
        clean_df['instrument'] = ticker
        clean_df['open'] = stock_data['Open']
        clean_df['high'] = stock_data['High']
        clean_df['low'] = stock_data['Low']
        clean_df['close'] = stock_data['Close']
        # Use .get() for 'Volume' as it might not exist for indices like ^NSEI
        clean_df['volume'] = stock_data.get('Volume')

        return clean_df

    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def run_collector(sheet):
    """Main function to run data collection and upload to Google Sheets."""
    all_data = [fetch_historical_data(symbol) for symbol in SYMBOLS]
    
    # Filter out any empty DataFrames from failed fetches
    all_data = [df for df in all_data if not df.empty]

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Successfully fetched a total of {len(combined_df)} rows of data.")

        combined_df['timestamp'] = combined_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Replace any NaN (Not a Number) values with an empty string ('').
        # This is a robust way to make the data JSON-compliant for the API call.
        combined_df.fillna('', inplace=True)

        rows_to_append = combined_df.values.tolist()
        data_to_write = [SHEET_HEADERS] + rows_to_append

        print(f"Preparing to write {len(data_to_write)} rows (including header).")
        # Print a sample of the data to be written for debugging
        if len(data_to_write) > 1:
            print("Sample data row: ", data_to_write[1])

        try:
            print("Writing data to Google Sheet...")
            # Clear the sheet first, then update it with all new data.
            sheet.clear()
            # The 'USER_ENTERED' option helps Google Sheets interpret the data correctly.
            sheet.update(data_to_write, value_input_option='USER_ENTERED')
            
            print("\n✅ Data written successfully!")
            print("IMPORTANT: Please REFRESH your browser tab with the Google Sheet to see the changes.")

        except Exception as e:
            print(f"\n❌ AN ERROR OCCURRED WHILE WRITING TO THE SHEET ❌")
            print(f"Error details: {e}")
            print("\nThis almost always means there is a PERMISSIONS problem.")
            print("Please double-check that this exact email address has the 'Editor' role in your Google Sheet's 'Share' settings:")
            print(f"   -> {sheet.client.auth.service_account_email}")
    else:
        print("No data was fetched. Please check symbols or internet connection.")


if __name__ == "__main__":
    print("Starting data collection...")
    google_sheet = connect_to_google_sheets()
    run_collector(google_sheet)