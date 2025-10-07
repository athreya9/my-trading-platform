import requests
import os
from datetime import datetime
import time # Added import for time.sleep

# It is recommended to store the API key as an environment variable or a GitHub secret.
# NEWS_API_KEY = os.environ.get("NEWS_API_KEY") # This will be replaced by multiple keys

class RateLimitError(Exception):
    """Custom exception for News API rate limit."""
    pass

def fetch_news_sentiment(stock_name):
    """
    Temporarily disabled to prevent live bot from crashing.
    Always returns "neutral".
    """
    return "neutral"