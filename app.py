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

app = Flask(__name__)

# Enable CORS to allow the frontend to make requests to this API
CORS(app)

# Register the API endpoints from process_data.py under the /api prefix
app.register_blueprint(process_data_bp, url_prefix='/api')

@app.route('/')
def index():
    """A simple health-check endpoint to confirm the server is running."""
    # Triggering a new deployment
    return "Python backend is running."

if __name__ == '__main__':
    # This block is for local development and is not used by Gunicorn in production
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)