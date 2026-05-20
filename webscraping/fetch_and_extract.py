# fetch_and_extract.py
import requests
import time
import pandas as pd
from newspaper import Article
from dotenv import load_dotenv
import os

# === CONFIGURATION ===
load_dotenv()
API_KEY = os.getenv("GNEWS_API_KEY")  # Replace with your actual API key
TARGET_COUNT = 1000            # Target number of articles
MAX_PER_PAGE = 100             # Max articles per request (API limit)
BASE_URL = "https://gnews.io/api/v4/search"

# Economic search query (URL-encoded)
# Searches for titles/descriptions containing these economic terms
QUERY = '(economy OR economic OR inflation OR unemployment OR GDP OR "interest rate" OR "central bank" OR "fiscal policy" OR "monetary policy" OR "trade deficit")'

# === MAIN FUNCTION ===
def fetch_and_extract():
    all_articles = []
    page = 1
    
    print(f"Targeting {TARGET_COUNT} articles about economic news in Malaysia...")
    
    while len(all_articles) < TARGET_COUNT:
        print(f"\n--- Fetching page {page} (offset: {MAX_PER_PAGE * (page - 1)}) ---")
        
        # Build request parameters
        params = {
            'q': QUERY,
            'lang': 'en',
            'country': 'my',  # Malaysia
            'max': MAX_PER_PAGE,
            'page': page,
            'apikey': API_KEY
        }
        
        # Make the API request
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            break
        
        # Check if articles were returned
        articles = data.get('articles', [])
        if not articles:
            print("No more articles found. Stopping pagination.")
            break
        
        print(f"Retrieved {len(articles)} articles. Processing...")
        
        # Process each article in the batch
        for item in articles:
            title = item.get('title', 'No title')
            url = item.get('url')
            
            print(f"  Extracting: {title[:60]}...")
            
            # Use newspaper3k to get full article text
            article = Article(url)
            try:
                article.download()
                article.parse()
                article.nlp()
                
                all_articles.append({
                    'title': title,
                    'url': url,
                    'description': item.get('description', ''),
                    'publishedAt': item.get('publishedAt', ''),
                    'source': item.get('source', {}).get('name', ''),
                    'text': article.text,
                    'summary': article.summary,
                    'keywords': article.keywords
                })
                print(f"    ✓ Successfully extracted ({len(all_articles)}/{TARGET_COUNT})")
            except Exception as e:
                print(f"    ✗ Failed to extract article: {e}")
                # Still add metadata even if full text fails
                all_articles.append({
                    'title': title,
                    'url': url,
                    'description': item.get('description', ''),
                    'publishedAt': item.get('publishedAt', ''),
                    'source': item.get('source', {}).get('name', ''),
                    'text': None,
                    'summary': None,
                    'keywords': None
                })
            
            # Save checkpoint every 200 articles
            if len(all_articles) % 200 == 0:
                temp_df = pd.DataFrame(all_articles)
                temp_df.to_csv('economic_news_checkpoint.csv', index=False)
                print(f"  💾 Checkpoint saved: {len(all_articles)} articles so far")
            
            # Stop if we've reached the target
            if len(all_articles) >= TARGET_COUNT:
                break
            
            # Polite delay between article extractions (1 second)
            time.sleep(1)
        
        # Move to next page
        page += 1
        
        # Polite delay between API requests (2 seconds)
        time.sleep(2)
        
        # Safety: prevent infinite loops if API doesn't paginate correctly
        if page > 50:  # Max 50 pages * 100 articles = 5000 articles
            print("Reached maximum page limit. Stopping.")
            break
    
    # Save final dataset
    final_df = pd.DataFrame(all_articles[:TARGET_COUNT])
    final_df.to_csv('malaysian_economic_news_1000.csv', index=False)
    print(f"\n✅ Done! Saved {len(final_df)} articles to 'malaysian_economic_news_1000.csv'")

if __name__ == '__main__':
    fetch_and_extract()
