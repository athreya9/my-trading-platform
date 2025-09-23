#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import firestore
import os
import logging

# --- Setup ---
logger = logging.getLogger("uvicorn")
app = FastAPI()

# --- CORS Configuration ---
# This is crucial to allow your frontend Cloud Run service to access this API.
origins = [
    "http://localhost:3000",  # For local Next.js development
    # The public URL of your 'trading-platform-analysis-dashboard' Cloud Run service
    "https://trading-platform-analysis-dashboard-884404713353.us-west1.run.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Firestore Initialization ---
try:
    # In a Google Cloud environment (like Cloud Run), credentials are automatically
    # discovered if the service account has the correct permissions (e.g., Cloud Datastore User).
    firebase_admin.initialize_app()
    db = firestore.client()
    logger.info("Successfully initialized Firestore client.")
except Exception as e:
    logger.error(f"Failed to initialize Firestore: {e}")
    # This is a fatal error for the app, but we'll let endpoints handle failures.
    db = None

# --- API Endpoints ---
@app.get("/api/health")
def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok", "firestore_initialized": db is not None}

@app.get("/api/trading-data")
def get_trading_data():
    """
    Fetches trading data from a Firestore collection.
    This endpoint queries a collection and returns the documents.
    """
    if db is None:
        raise HTTPException(status_code=500, detail="Server configuration error: Firestore client not initialized.")

    try:
        # IMPORTANT: Replace 'pnl_history' with your actual Firestore collection name.
        collection_ref = db.collection('pnl_history').order_by('Date', direction=firestore.Query.DESCENDING).limit(100)
        docs = collection_ref.stream()

        # Convert documents to a list of dictionaries
        data = [doc.to_dict() for doc in docs]

        if not data:
            return {"message": "No data found in the collection.", "data": []}

        return {"data": data}
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching data from Firestore: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while fetching data.")