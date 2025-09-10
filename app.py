#!/usr/bin/env python3
"""
A Flask web server to provide API endpoints for controlling the trading bot.
"""
import gspread
from flask import Flask, request, jsonify
import os
import json

# Import the blueprint
from api.process_data import process_data_bp

app = Flask(__name__)

# Register the blueprint from api/process_data.py
app.register_blueprint(process_data_bp)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'api', 'config.py')

@app.route('/')
def index():
    """A simple welcome message to confirm the app is running."""
    return jsonify({
        "status": "online",
        "message": "Welcome to the Trading Bot API.",
        "endpoints": [
            "/run?force=true",
            "/api/switch-mode (POST)",
            "/api/bot-status (POST)"
        ]
    })

@app.route('/api/switch-mode', methods=['POST'])
def switch_mode():
    """
    Switches the trading mode in the config file between 'FULL' and 'EMERGENCY'.
    This endpoint safely reads the config file, replaces the mode, and writes it back.
    """
    try:
        new_mode = request.json.get('mode', 'EMERGENCY').upper()

        if new_mode not in ['FULL', 'EMERGENCY']:
            return jsonify({'status': 'error', 'message': "Invalid mode. Must be 'FULL' or 'EMERGENCY'."}), 400

        with open(CONFIG_PATH, 'r') as f:
            lines = f.readlines()

        updated_lines = []
        found = False
        for line in lines:
            if line.strip().startswith('CURRENT_MODE ='):
                updated_lines.append(f"CURRENT_MODE = '{new_mode}'\n")
                found = True
            else:
                updated_lines.append(line)
        
        if not found:
            return jsonify({'status': 'error', 'message': "Could not find 'CURRENT_MODE' variable in config file."}), 500

        with open(CONFIG_PATH, 'w') as f:
            f.writelines(updated_lines)

        return jsonify({'status': 'success', 'mode': new_mode})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/bot-status', methods=['POST'])
def bot_status():
    """
    Sets the bot's status in the Google Sheet's 'Bot_Control' tab.
    """
    try:
        new_status = request.json.get('status', 'running').lower()
        if new_status not in ['running', 'stopped']:
            return jsonify({'status': 'error', 'message': "Invalid status. Must be 'running' or 'stopped'."}), 400

        # Authenticate with Google Sheets using the environment variable
        creds_json_str = os.getenv('GOOGLE_SHEETS_CREDENTIALS')
        creds_dict = json.loads(creds_json_str)
        client = gspread.service_account_from_dict(creds_dict)
        spreadsheet = client.open("Algo Trading Dashboard")
        worksheet = spreadsheet.worksheet("Bot_Control")

        # Update the status cell (assuming cell B2 corresponds to 'status')
        # This overwrites the status to either "running" or "stopped"
        worksheet.update_acell('B2', new_status)

        return jsonify({'status': 'success', 'message': f"Bot status set to '{new_status}'."})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # This block is for running the app in a container like Cloud Run
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)