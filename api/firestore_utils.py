import firebase_admin
from firebase_admin import credentials, firestore
import logging

logger = logging.getLogger(__name__)

def init_firestore_client():
    """
    Initializes the Firestore client using service account credentials.
    Ensures the app is initialized only once.
    """
    try:
        if not firebase_admin._apps:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {'projectId': 'my-trading-platform-471103'})
            logger.info("Successfully initialized Firebase Admin SDK.")
        else:
            logger.info("Firebase Admin SDK already initialized (default app).")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}", exc_info=True)
        raise

def get_db():
    """
    Returns a Firestore client instance.
    Ensures Firebase app is initialized before getting the client.
    """
    if not firebase_admin._apps:
        logger.warning("Firebase app not initialized when get_db() was called. Attempting to initialize.")
        init_firestore_client()

    try:
        db = firestore.client()
        logger.info("Firestore client obtained successfully.")
        return db
    except Exception as e:
        logger.error(f"Failed to get Firestore client: {e}", exc_info=True)
        raise

# You might also need read_collection and write_data_to_firestore if they are used elsewhere
def write_data_to_firestore(db, collection_name, data):
    """
    Writes a list of dictionaries to a Firestore collection.
    Each dictionary will be a new document in the collection.
    """
    try:
        collection_ref = db.collection(collection_name)
        for item in data:
            collection_ref.add(item)
        logger.info(f"Successfully wrote {len(data)} documents to collection '{collection_name}'.")
    except Exception as e:
        logger.error(f"Failed to write to collection '{collection_name}': {e}", exc_info=True)
        raise

def read_collection(db, collection_name):
    """
    Reads all documents from a Firestore collection.
    """
    try:
        collection_ref = db.collection(collection_name)
        docs = collection_ref.stream()
        data = [doc.to_dict() for doc in docs]
        logger.info(f"Successfully read {len(data)} documents from collection '{collection_name}'.")
        return data
    except Exception as e:
        logger.error(f"Failed to read from collection '{collection_name}': {e}", exc_info=True)
        raise
