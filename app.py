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

    # Enable CORS to allow the frontend to make requests to this API
    CORS(app)
    logging.info("CORS enabled.")

    # Register the API endpoints from process_data.py under the /api prefix
    app.register_blueprint(process_data_bp, url_prefix='/api')
    logging.info("Blueprint registered.")
except Exception as e:
    logging.error(f"Error initializing Flask app: {e}", exc_info=True)
    raise

@app.route('/')
def index():
    logging.info("Health check endpoint was hit")
    """A simple health-check endpoint to confirm the server is running."""
    # Triggering a new deployment
    logging.info("Returning health check message")
    return "Python backend is running."

if __name__ == '__main__':
    logging.info("Running Flask app in main block (local development).")
    # This block is for local development and is not used by Gunicorn in production
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Setting port to: {port}")
    logging.info("Starting app.run...")
    app.run(host='0.0.0.0', port=port, debug=True)
    logging.info("app.run completed.")