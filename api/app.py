from flask import Flask
import os
from flask_cors import CORS
from process_data import process_data_bp

app = Flask(__name__)

# Enable CORS for all domains on all routes.
# This allows your frontend (on localhost:3000) to talk to your backend (on localhost:8080).
CORS(app)

# Register the blueprint from process_data.py
# All routes in that file will now be available under the /api prefix.
app.register_blueprint(process_data_bp, url_prefix='/api')

@app.route('/')
def index():
    return "Python backend is running."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)