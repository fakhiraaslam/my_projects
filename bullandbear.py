import datetime
from newsapi import NewsApiClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dateutil import rrule
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

newsapi = NewsApiClient(api_key='f8e4a91e06dd47e4a3c1055ca58df828')

def get_news(stock_name, start_date, end_date):
    articles_list = []
    for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        day_str = dt.strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q=stock_name, from_param=day_str, to=day_str, language='en', sort_by='relevancy')
        articles_list.extend(articles['articles']) 
    return articles_list

# Example usage for Apple from September 25, 2024 to October 24, 2024
start_date = datetime.datetime(2024, 9, 25)
end_date = datetime.datetime(2024, 10, 24)
articles = get_news('Apple', start_date, end_date)

print(f"Number of articles found: {len(articles)}")

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(articles):
    sentiment_scores = []
    for article in articles:
        sentiment = sia.polarity_scores(article['description'] if article['description'] else article['title'])
        sentiment_scores.append(sentiment['compound']) 
    return sentiment_scores

sentiment_scores = analyze_sentiment(articles)
print(sentiment_scores[:5])

def get_stock_data(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(start=start_date, end=end_date)
    return data

# Fetch historical stock data for the same date range
stock_data = get_stock_data('AAPL', start_date, end_date)
print(stock_data)

# Create the dataset using sentiment scores and percentage stock price change
def create_dataset_for_regression(sentiment_scores, stock_data):
    # Sentiment scores as features (X)
    X = np.array(sentiment_scores[:-1])  # Use sentiment scores for days 0 to n-1
    # Percentage change in closing price as target (y)
    y = calculate_price_change(stock_data)  # Use price change from days 1 to n
    return X, y

def calculate_price_change(stock_data):
    close_prices = stock_data['Close'].values
    # Calculate percentage change in closing prices
    price_change = (close_prices[1:] - close_prices[:-1]) / close_prices[:-1]
    return price_change

# Create the dataset using sentiment scores and percentage stock price change
def create_dataset_for_regression(sentiment_scores, stock_data):
    # Ensure the sentiment scores and stock data align correctly
    num_samples = min(len(sentiment_scores), len(stock_data['Close']) - 1)
    
    # Use sentiment scores for days 0 to num_samples-1
    X = np.array(sentiment_scores[:num_samples])
    
    # Calculate percentage change in closing price as target (y)
    y = calculate_price_change(stock_data)[:num_samples]
    
    return X, y

# Example usage (assuming sentiment_scores and stock_data are already fetched)
X, y = create_dataset_for_regression(sentiment_scores, stock_data)

# Reshape X to 2D array (n_samples, 1 feature) for linear regression
X = X.reshape(-1, 1)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Test the model (predictions for X_test)
predictions = model.predict(X_test)

# Sort the test set for better visualization
sorted_idx = np.argsort(X_test[:, 0])
X_test_sorted = X_test[sorted_idx]
predictions_sorted = predictions[sorted_idx]
y_test_sorted = y_test[sorted_idx]

# Plot the graph
plt.figure(figsize=(10, 6))

# Plot actual stock price percentage change
plt.scatter(X_test_sorted, y_test_sorted, color='blue', label='Actual Stock Price Change')

# Plot predicted stock price percentage change (using the linear regression model)
plt.plot(X_test_sorted, predictions_sorted, color='red', label='Predicted Stock Price Change', linewidth=2)

plt.xlabel('Sentiment Score')
plt.ylabel('Stock Price Change (%)')
plt.title('Linear Regression: Sentiment Score vs Stock Price Change')
plt.legend()

# Show the plot
plt.show()