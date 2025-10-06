import requests
import os
from datetime import datetime
import time # Added import for time.sleep

# It is recommended to store the API key as an environment variable or a GitHub secret.
# NEWS_API_KEY = os.environ.get("NEWS_API_KEY") # This will be replaced by multiple keys

class RateLimitError(Exception):
    """Custom exception for News API rate limit."""
    pass

def fetch_news_sentiment(stock_name): # Modified to accept api_key
    api_keys = [os.getenv(f"NEWS_API_KEY_{i}") for i in range(1, 4)] # Assuming NEWS_API_KEY_1, NEWS_API_KEY_2, NEWS_API_KEY_3
    api_keys = [key for key in api_keys if key] # Filter out None values

    if not api_keys:
        print("No NEWS_API_KEYs are set. Returning neutral sentiment.")
        log_file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'sentiment_fallback.log')
        with open(log_file_path, "a") as log:
            log.write(f"{datetime.now()} - Fallback used for {stock_name} (No API keys)\n")
        return "neutral" # Changed return type

    for key in api_keys:
        url = f"https://newsapi.org/v2/everything?q={stock_name}&language=en&sortBy=publishedAt&pageSize=10&apiKey={key}"
        try:
            response = requests.get(url)
            if response.status_code == 429:
                print(f"⚠️ News API rate limit hit for key {key[-4:]}. Trying next key.")
                log_file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'sentiment_fallback.log')
                with open(log_file_path, "a") as log:
                    log.write(f"{datetime.now()} - Rate limit hit for key {key[-4:]} for {stock_name}. Trying next.\n")
                raise RateLimitError # Raise custom error to catch and try next key
            response.raise_for_status() # Raise an exception for other bad status codes
            articles = response.json().get("articles", [])

            if not articles:
                return "neutral" # Changed return type

            # Simple keyword-based sentiment scoring
            positive_keywords = ["gain", "profit", "growth", "surge", "bullish", "strong", "record", "hike", "upgrade", "buy"]
            negative_keywords = ["loss", "fall", "drop", "bearish", "weak", "decline", "cut", "miss", "downgrade", "sell"]

            score = 0
            # summary = [] # Removed summary as return type is now string

            for article in articles:
                title = article.get("title", "").lower()
                description = article.get("description", "").lower()
                content = f"{title} {description}"

                pos_hits = sum(word in content for word in positive_keywords)
                neg_hits = sum(word in content for word in negative_keywords)
                score += pos_hits - neg_hits

                # if article.get("title"): # Removed summary as return type is now string
                #     summary.append(f"- {article.get('title')}")

            # Convert score to sentiment string
            if score > 0:
                return "positive"
            elif score < 0:
                return "negative"
            else:
                return "neutral"

        except RateLimitError: # Catch custom error to continue to next key
            continue
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for {stock_name} with key {key[-4:]}: {e}")
            # Don't return immediately, try next key if available
        finally:
            time.sleep(1) # Sleep for 1 second after each call

    # If all keys fail or no keys are available
    print(f"⚠️ News API rate limit hit for all keys or other error. Using fallback sentiment for {stock_name}.")
    log_file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'sentiment_fallback.log')
    with open(log_file_path, "a") as log:
        log.write(f"{datetime.now()} - Fallback used for {stock_name} (All keys failed or no keys)\n")
    return "neutral" # Changed return type