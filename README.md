
# Stock Market News Sentiment Analyzer

This project analyzes the relationship between news sentiment and stock price changes for a given company. It fetches relevant news articles, calculates sentiment scores, and correlates them with historical stock price movements using linear regression.

## Features
- Fetches news articles for a stock symbol using **NewsAPI**.
- Analyzes article sentiment using **VADER Sentiment Analysis**.
- Retrieves historical stock data using **Yahoo Finance**.
- Predicts stock price movements based on sentiment scores with **Linear Regression**.
- Visualizes actual vs. predicted stock price changes.

## Status and Future Work
This project is a work in progress, and I'm continuously working on it to add new features, optimize performance, and improve prediction accuracy. Stay tuned for future updates!

## Requirements
- Python 3.x
- NewsAPI key (sign up at [NewsAPI](https://newsapi.org/))
- Required libraries (install using `pip`):
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/stock-news-sentiment-analyzer.git
    cd stock-news-sentiment-analyzer
    ```

2. Add your **NewsAPI** key in `config.py`:
    ```python
    NEWS_API_KEY = 'your_api_key_here'
    ```

3. Run the script:
    ```bash
    python main.py
    ```

## Example
Analyze news sentiment for **Apple (AAPL)** and its correlation with stock price changes from September 25, 2024, to October 24, 2024:
```python
start_date = datetime.datetime(2024, 9, 25)
end_date = datetime.datetime(2024, 10, 24)
articles = get_news('Apple', start_date, end_date)
```
