# fetch_and_extract.py
import requests
import time
import pandas as pd
from newspaper import Article
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import os

# === CONFIGURATION ===
load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")  # Get free key at https://newsapi.org
TARGET_COUNT = 1000
PAGE_SIZE = 100                     # NewsAPI max per request
BASE_URL = "https://newsapi.org/v2/everything"

# Last 30 days (NewsAPI free tier limit)
FROM_DATE = (datetime.now(timezone.utc) - timedelta(days=29)).strftime("%Y-%m-%dT%H:%M:%SZ")
TO_DATE   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

QUERIES = [
    'inflation rate economic growth outlook',
    'interest rate hike recession unemployment',
    'GDP growth central bank policy',
    'trade war tariffs supply chain jobs',
    'stock market rally inflation earnings',
    'federal reserve rate decision employment',
    'housing market mortgage rates affordability',
    'oil price inflation consumer spending',
    'tech layoffs economic slowdown investment',
    'currency devaluation exports trade balance',
    'budget deficit government spending stimulus',
    'wage growth inflation purchasing power',
    'supply chain disruption economic recovery',
    'emerging markets capital outflows dollar',
    'corporate earnings recession consumer demand',
]

# === HELPERS ===

def fetch_articles_for_query(query: str, seen_urls: set) -> list:
    """Page through NewsAPI for one query, returning deduplicated articles."""
    results = []
    page = 1

    while True:
        params = {
            'q': query,
            'language': 'en',
            'pageSize': PAGE_SIZE,
            'page': page,
            'from': FROM_DATE,
            'to': TO_DATE,
            'sortBy': 'relevancy',
            'apiKey': API_KEY,
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=15)
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"    Request failed (page {page}): {e}")
            break

        if data.get('status') != 'ok':
            print(f"    API error: {data.get('message', data)}")
            break

        articles = data.get('articles', [])
        total = data.get('totalResults', 0)

        if not articles:
            break

        new_this_page = 0
        for item in articles:
            url = item.get('url')
            if not url or url in seen_urls or url == 'https://removed.com':
                continue
            seen_urls.add(url)
            results.append(item)
            new_this_page += 1

        print(f"    Page {page}: {len(articles)} returned, {new_this_page} new (total available: {total})")

        # Stop if we've seen all results or hit NewsAPI free tier page cap (page 1-10 only on free)
        if page >= 10 or page * PAGE_SIZE >= total or len(articles) < PAGE_SIZE:
            break

        page += 1
        time.sleep(1)

    return results


def extract_full_text(item: dict) -> dict:
    """Download and parse full article text with newspaper3k."""
    url = item.get('url')
    title = item.get('title', 'No title')
    print(f"  Extracting: {title[:70]}...")

    article = Article(url)
    try:
        article.download()
        article.parse()
        article.nlp()
        return {
            'title': title,
            'url': url,
            'description': item.get('description', ''),
            'publishedAt': item.get('publishedAt', ''),
            'source': item.get('source', {}).get('name', ''),
            'text': article.text,
            'summary': article.summary,
            'keywords': article.keywords,
        }
    except Exception as e:
        print(f"    ✗ Full-text extraction failed: {e}")
        return {
            'title': title,
            'url': url,
            'description': item.get('description', ''),
            'publishedAt': item.get('publishedAt', ''),
            'source': item.get('source', {}).get('name', ''),
            'text': None,
            'summary': None,
            'keywords': None,
        }


# === MAIN ===

def fetch_and_extract():
    seen_urls: set = set()
    raw_items: list = []

    print(f"Targeting {TARGET_COUNT} articles (date range: {FROM_DATE} → {TO_DATE})")
    print(f"Running {len(QUERIES)} sub-queries...\n")

    for i, query in enumerate(QUERIES, 1):
        if len(raw_items) >= TARGET_COUNT:
            break
        print(f"[{i}/{len(QUERIES)}] Query: {query!r}")
        batch = fetch_articles_for_query(query, seen_urls)
        raw_items.extend(batch)
        print(f"  → {len(batch)} new articles (total so far: {len(raw_items)})\n")
        time.sleep(1)

    print(f"\nCollected {len(raw_items)} unique articles.")
    print(f"Extracting full text for up to {TARGET_COUNT}...\n")

    all_articles = []
    for idx, item in enumerate(raw_items[:TARGET_COUNT], 1):
        record = extract_full_text(item)
        all_articles.append(record)
        print(f"    ✓ ({idx}/{min(len(raw_items), TARGET_COUNT)})")

        if idx % 200 == 0:
            pd.DataFrame(all_articles).to_csv('economic_news_checkpoint.csv', index=False)
            print(f"  💾 Checkpoint saved at {idx} articles")

        time.sleep(1)

    final_df = pd.DataFrame(all_articles)
    out_file = 'economic_news_1000.csv'
    final_df.to_csv(out_file, index=False)
    print(f"\n✅ Done! Saved {len(final_df)} articles to '{out_file}'")
    print(f"   Date range in dataset: {final_df['publishedAt'].min()}  →  {final_df['publishedAt'].max()}")


if __name__ == '__main__':
    fetch_and_extract()
