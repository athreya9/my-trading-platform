#!/usr/bin/env python3
"""
Main Flask application file.
This acts as the single entry point for the Gunicorn server.
"""
import os
from flask import Flask
from flask_cors import CORS
import logging

# Import the blueprint from the api package
from api.process_data import process_data_bp

# --- Standard Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s UTC - %(levelname)s - %(message)s')

def create_app():
    """
    Application factory to create and configure the Flask app.
    This pattern is robust and avoids circular imports.
    """
    app = Flask(__name__)
    logging.info("Flask app created.")

    # --- Definitive CORS Configuration ---
    # This is the critical fix for connecting the frontend to the backend.
    # It explicitly tells the browser that requests from your deployed frontend
    # URL are permitted.
    origins = [
        "https://trading-platform-analysis-dashboard-884404713353.us-west1.run.app",  # Your NEW production frontend
        "http://localhost:3000",                      # For local React dev server
        "http://127.0.0.1:3000"                       # Alternative for local dev
    ]
    CORS(app, origins=origins, supports_credentials=True)
    logging.info(f"CORS enabled for all routes with origins: {origins}")

    # Register the API endpoints from process_data.py under the /api prefix
    app.register_blueprint(process_data_bp, url_prefix='/api')
    logging.info("API blueprint registered under /api prefix.")

    @app.route('/')
    def index():
        """A simple health-check endpoint to confirm the server is running."""
        # Triggering a new deployment.
        return "Python backend is running."

    return app

# Create the app instance for Gunicorn to use
app = create_app()

if __name__ == '__main__':
    # This block is for local development and is not used by Gunicorn in production.
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Starting Flask app for local development on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)