#!/usr/bin/env python3
"""
Main Flask application file.
This acts as the single entry point for the Gunicorn server.
"""
import os
from flask import Flask
from flask_cors import CORS

# Import the blueprint
from api.process_data import process_data_bp

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting Flask application...")

try:
    app = Flask(__name__)
    logging.info("Flask app created.")

    # --- Enable CORS for the specific frontend origin ---
    # This is a critical security and functionality step. It tells the browser
    # that it's safe to allow the frontend code to access this backend API.
    # We target the /api/* routes and specify the exact URL of the deployed frontend.
    frontend_url = "https://my-trading-platform-471103.web.app"
    CORS(app, resources={r"/api/*": {"origins": frontend_url}})
    logging.info("CORS enabled.")

    # Register the API endpoints from process_data.py under the /api prefix
    app.register_blueprint(process_data_bp, url_prefix='/api')
    logging.info("Blueprint registered.")
except Exception as e:
    logging.error(f"Error initializing Flask app: {e}", exc_info=True)
    raise

@app.route('/')
def index():
    """A simple health-check endpoint to confirm the server is running."""
    return "Python backend is running."

if __name__ == '__main__':
    logging.info("Running Flask app in main block (local development).")
    # This block is for local development and is not used by Gunicorn in production
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Setting port to: {port}")
    logging.info("Starting app.run...")
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False for production-like testing
    logging.info("app.run completed.")
