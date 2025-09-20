#!/usr/bin/env python3
"""
Main Flask application file.
This acts as the single entry point for the Gunicorn server.
"""
import os
from flask import Flask
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_app():
    """
    Application factory to create and configure the Flask app.
    This pattern prevents import-time errors and makes the app more modular.
    """
    logging.info("Creating Flask application...")
    app = Flask(__name__)

    # Configure CORS to allow requests from your frontend.
    # Using "*" is acceptable for now, but for production, you might restrict
    # it to your specific frontend URL.
    CORS(app, resources={r"/api/*": {
        "origins": "*",
        "allow_headers": ["Content-Type", "Authorization"]
    }})
    logging.info("CORS enabled.")

    # --- Import and Register Blueprints ---
    # We import here to avoid circular dependencies and ensure the app context is available.
    from api.process_data import process_data_bp
    app.register_blueprint(process_data_bp, url_prefix='/api')
    logging.info("API blueprint registered.")

    @app.route('/')
    def index():
        """A simple health-check endpoint to confirm the server is running."""
        return "Python backend is running."

    return app

# Create the app instance for Gunicorn to use
app = create_app()

if __name__ == '__main__':
    logging.info("Running Flask app in main block (local development).")
    # This block is for local development and is not used by Gunicorn in production
    port = int(os.environ.get("PORT", 8080))
    logging.info(f"Setting port to: {port}")
    logging.info("Starting app.run...")
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=False for production-like testing
    logging.info("app.run completed.")
