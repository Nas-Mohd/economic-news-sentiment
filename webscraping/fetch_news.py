# fetch_news.py
from gnews import GNews

# Initialize the GNews client for Malaysian business news
google_news = GNews(language='en', country='MY', max_results=10)
google_news.period = '7d'  # Fetch news from the last 7 days

# Get top headlines
top_headlines = google_news.get_top_news()

# Print the results
if top_headlines:
    for idx, article in enumerate(top_headlines):
        print(f"{idx + 1}. {article['title']}")
        print(f"   URL: {article['url']}\n")
else:
    print("No news articles found.")