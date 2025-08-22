import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# -------------------------------
# Configuration
# -------------------------------
NEWS_API_KEY = "1675e649560e401c852bac821b1cf85d"
STOCK_SYMBOL = "SPY"  # S&P500 ETF as proxy
OUTPUT_DIR = "outputs"
DATA_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


analyzer = SentimentIntensityAnalyzer()
newsapi = NewsApiClient(api_key=NEWS_API_KEY)


# -------------------------------
# Shock dates (example)
# -------------------------------
shock_dates_list = ["2025-07-17", "2025-07-20", "2025-08-01"]  # Add more as needed
shock_dates_csv = os.path.join(DATA_DIR, "shock_dates.csv")
pd.DataFrame({'Date': shock_dates_list}).to_csv(shock_dates_csv, index=False)


# Earliest date allowed on free NewsAPI plan - updated as per API error message
FREE_PLAN_START_DATE = datetime.strptime("2025-07-21", "%Y-%m-%d")


def fetch_news(symbol, from_date, to_date, page_size=100):
    """Fetch news articles from NewsAPI."""
    all_articles = newsapi.get_everything(
        q=symbol,
        from_param=from_date,
        to=to_date,
        language='en',
        sort_by='relevancy',
        page_size=page_size
    )
    return all_articles['articles']


def analyze_sentiment(text):
    """Return compound sentiment score using VADER."""
    if not text:
        return 0
    return analyzer.polarity_scores(text)['compound']


def run_model(days_before_after=3):
    """
    Runs sentiment analysis for shock dates within NewsAPI free plan range.
    Automatically clamps from_date to the earliest allowed date.
    """
    df_shocks = pd.read_csv(shock_dates_csv)
    results = []

    for shock_date_str in df_shocks['Date']:
        shock_date = datetime.strptime(shock_date_str, "%Y-%m-%d")

        # Skip shock dates outside the free plan range
        if shock_date < FREE_PLAN_START_DATE:
            print(f"Skipping {shock_date_str}: outside NewsAPI free plan range")
            continue

        # Clamp from_date to the earliest allowed date, strictly after FREE_PLAN_START_DATE
        from_dt = max(shock_date - timedelta(days=days_before_after), FREE_PLAN_START_DATE)
        if from_dt <= FREE_PLAN_START_DATE:
            from_dt = FREE_PLAN_START_DATE + timedelta(days=1)

        to_dt = shock_date + timedelta(days=days_before_after)

        from_date = from_dt.strftime("%Y-%m-%d")
        to_date = to_dt.strftime("%Y-%m-%d")

        print(f"Fetching news for {shock_date_str}: from {from_date} to {to_date}")

        articles = fetch_news(STOCK_SYMBOL, from_date, to_date)
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            text = " ".join(filter(None, [title, description, content]))
            score = analyze_sentiment(text)
            results.append({
                'Shock_Date': shock_date_str,
                'Published_At': article['publishedAt'],
                'Title': title,
                'Score': score
            })

    result_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "sentiment_scores.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Sentiment CSV saved at: {csv_path}")

    # Plot average sentiment per shock date
    plt.figure(figsize=(10,6))
    avg_scores = result_df.groupby('Shock_Date')['Score'].mean()
    avg_scores.plot(kind='bar', color='purple')
    plt.xlabel("Shock Date")
    plt.ylabel("Average Sentiment Score")
    plt.title("Sentiment Analysis Around Shocks")
    plt.grid(True)
    plot_path = os.path.join(OUTPUT_DIR, "sentiment_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Sentiment plot saved at: {plot_path}")
