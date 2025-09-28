
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()

class DataFetcher:
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API')
        self.news_api_key = os.getenv('NEWS_API')
        self.base_url_av = "https://www.alphavantage.co/query"
        self.base_url_news = "https://newsapi.org/v2/everything"

        if not self.alpha_vantage_key or not self.news_api_key:
            print("Warning: API keys not found in environment variables")

    def get_stock_data(self, symbol):
        """Fetch real-time stock data from Alpha Vantage"""
        try:
            # Get global quote (real-time price)
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }

            response = requests.get(self.base_url_av, params=params)
            data = response.json()

            if "Global Quote" in data:
                quote = data["Global Quote"]
                return {
                    'symbol': symbol,
                    'price': float(quote['05. price']),
                    'change': float(quote['09. change']),
                    'change_percent': float(quote['10. change percent'].rstrip('%')),
                    'volume': int(quote['06. volume']),
                    'high': float(quote['03. high']),
                    'low': float(quote['04. low']),
                    'open': float(quote['02. open']),
                    'previous_close': float(quote['08. previous close']),
                    'timestamp': datetime.now()
                }
            else:
                print(f"Error fetching data for {symbol}: {data}")
                return self._get_mock_stock_data(symbol)

        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return self._get_mock_stock_data(symbol)

    def get_intraday_data(self, symbol, interval='5min'):
        """Fetch intraday data for technical analysis"""
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.alpha_vantage_key
            }

            response = requests.get(self.base_url_av, params=params)
            data = response.json()

            if f"Time Series ({interval})" in data:
                time_series = data[f"Time Series ({interval})"]

                df_data = []
                for timestamp, values in time_series.items():
                    df_data.append({
                        'timestamp': pd.to_datetime(timestamp),
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(values['5. volume'])
                    })

                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)

                return df
            else:
                print(f"Error fetching intraday data: {data}")
                return self._get_mock_intraday_data(symbol)

        except Exception as e:
            print(f"Error fetching intraday data: {e}")
            return self._get_mock_intraday_data(symbol)

    def get_news_data(self, symbol, days_back=7):
        """Fetch news data from News API"""
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)

            params = {
                'q': f'{symbol} OR {self._get_company_name(symbol)}',
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'apiKey': self.news_api_key,
                'language': 'en',
                'pageSize': 20
            }

            response = requests.get(self.base_url_news, params=params)
            data = response.json()

            if response.status_code == 200 and 'articles' in data:
                articles = []
                for article in data['articles']:
                    articles.append({
                        'title': article['title'],
                        'description': article['description'],
                        'url': article['url'],
                        'publishedAt': article['publishedAt'],
                        'source': article['source']['name'],
                        'content': article.get('content', '')
                    })

                return articles
            else:
                print(f"Error fetching news data: {data}")
                return self._get_mock_news_data(symbol)

        except Exception as e:
            print(f"Error fetching news data: {e}")
            return self._get_mock_news_data(symbol)

    def _get_company_name(self, symbol):
        """Map stock symbols to company names for better news search"""
        company_mapping = {
            'AAPL': 'Apple',
            'GOOGL': 'Google Alphabet',
            'MSFT': 'Microsoft',
            'TSLA': 'Tesla',
            'AMZN': 'Amazon',
            'SPY': 'S&P 500',
            'NVDA': 'NVIDIA',
            'META': 'Meta Facebook'
        }
        return company_mapping.get(symbol, symbol)

    def _get_mock_stock_data(self, symbol):
        """Generate mock stock data when API fails"""
        base_prices = {
            'AAPL': 150, 'GOOGL': 2800, 'MSFT': 330,
            'TSLA': 880, 'AMZN': 3400, 'SPY': 420,
            'NVDA': 450, 'META': 320
        }

        base_price = base_prices.get(symbol, 100)
        change = np.random.uniform(-5, 5)

        return {
            'symbol': symbol,
            'price': base_price + change,
            'change': change,
            'change_percent': (change / base_price) * 100,
            'volume': np.random.randint(1000000, 50000000),
            'high': base_price + abs(change) + np.random.uniform(0, 3),
            'low': base_price + change - np.random.uniform(0, 3),
            'open': base_price + np.random.uniform(-2, 2),
            'previous_close': base_price,
            'timestamp': datetime.now()
        }

    def _get_mock_intraday_data(self, symbol):
        """Generate mock intraday data"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)

        timestamps = pd.date_range(start=start_time, end=end_time, freq='5min')
        base_price = self._get_mock_stock_data(symbol)['price']

        data = []
        current_price = base_price

        for ts in timestamps:
            change = np.random.normal(0, 0.5)
            current_price = max(10, current_price + change)

            high = current_price + abs(np.random.normal(0, 0.3))
            low = current_price - abs(np.random.normal(0, 0.3))
            open_price = current_price + np.random.normal(0, 0.2)
            volume = np.random.randint(10000, 100000)

            data.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': current_price,
                'volume': volume
            })

        df = pd.DataFrame(data, index=timestamps)
        return df

    def _get_mock_news_data(self, symbol):
        """Generate mock news data"""
        company_name = self._get_company_name(symbol)

        mock_articles = [
            {
                'title': f'{company_name} reports strong quarterly earnings',
                'description': f'{company_name} exceeded analyst expectations with robust revenue growth and improved margins.',
                'url': f'https://example.com/{symbol}-earnings',
                'publishedAt': datetime.now().isoformat(),
                'source': 'Financial News',
                'content': f'Detailed analysis of {company_name} financial performance...'
            },
            {
                'title': f'{company_name} announces strategic partnership',
                'description': f'{company_name} has entered into a new partnership to expand market reach.',
                'url': f'https://example.com/{symbol}-partnership',
                'publishedAt': (datetime.now() - timedelta(hours=6)).isoformat(),
                'source': 'Business Wire',
                'content': f'{company_name} strategic initiatives continue to drive growth...'
            },
            {
                'title': f'Analysts upgrade {company_name} stock rating',
                'description': f'Multiple analysts have raised their price targets for {company_name}.',
                'url': f'https://example.com/{symbol}-upgrade',
                'publishedAt': (datetime.now() - timedelta(hours=12)).isoformat(),
                'source': 'Market Watch',
                'content': f'Bullish sentiment surrounds {company_name} following recent developments...'
            }
        ]

        return mock_articles
