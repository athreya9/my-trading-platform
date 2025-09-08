#!/usr/bin/env python3
"""
A Flask web server to provide API endpoints for controlling the trading bot.
"""
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'api', 'config.py')

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

if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible on the local network
    app.run(host='0.0.0.0', port=5001, debug=True)