import sys
import os
from flask import Flask
from flask_cors import CORS

# Add the project root directory to the Python path.
# This allows us to use absolute imports like 'from api.process_data ...'
# which is a more robust pattern.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api.process_data import process_data_bp

app = Flask(__name__)

# Enable CORS for all domains on all routes.
# This allows your frontend (on localhost:3000) to talk to your backend (on localhost:8080).
CORS(app)

# Register the blueprint from process_data.py.
# All routes in that file (like /dashboard and /run) will now be available under the /api prefix.
app.register_blueprint(process_data_bp, url_prefix='/api')

@app.route('/')
def index():
    """A simple health-check endpoint to confirm the server is running."""
    return "Python backend is running."

if __name__ == '__main__':
    # Use 0.0.0.0 to make the server accessible from outside the container/VM
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)