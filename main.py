# main.py
import os
from flask import Flask

# Import the blueprints that contain your API endpoints
from api.process_data import process_data_bp

# Initialize the Flask application
app = Flask(__name__)

# Register the blueprints to make the endpoints available
app.register_blueprint(process_data_bp)

if __name__ == "__main__":
    # This block is for local development and is not used by Gunicorn in production
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)