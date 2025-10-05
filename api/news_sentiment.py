import requests
import os

# It is recommended to store the API key as an environment variable or a GitHub secret.
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

def fetch_news_sentiment(stock_name):
    if not NEWS_API_KEY: # Check if API key is actually available
        print("NEWS_API_KEY is not set. Returning neutral sentiment.")
        return {"score": 0, "summary": "NEWS_API_KEY is not set."}

    url = f"https://newsapi.org/v2/everything?q={stock_name}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 429:
            print("News API rate limit exceeded. Returning neutral sentiment.")
            return {"score": 0, "summary": "News API rate limit exceeded."}
        response.raise_for_status() # Raise an exception for other bad status codes
        articles = response.json().get("articles", [])

        if not articles:
            return {"score": 0, "summary": "No recent news found."}

        # Simple keyword-based sentiment scoring
        positive_keywords = ["gain", "profit", "growth", "surge", "bullish", "strong", "record", "hike", "upgrade", "buy"]
        negative_keywords = ["loss", "fall", "drop", "bearish", "weak", "decline", "cut", "miss", "downgrade", "sell"]

        score = 0
        summary = []

        for article in articles:
            title = article.get("title", "").lower()
            description = article.get("description", "").lower()
            content = f"{title} {description}"

            pos_hits = sum(word in content for word in positive_keywords)
            neg_hits = sum(word in content for word in negative_keywords)
            score += pos_hits - neg_hits

            if article.get("title"):
                summary.append(f"- {article.get('title')}")

        return {
            "score": score,
            "summary": "\n".join(summary[:3])  # Top 3 headlines
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {stock_name}: {e}")
        return {"score": 0, "summary": "Error fetching news."}
    finally:
        time.sleep(1) # Sleep for 1 second after each call