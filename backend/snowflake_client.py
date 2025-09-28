
import snowflake.connector
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import json

load_dotenv()

class SnowflakeClient:
    def __init__(self):
        self.account = os.getenv('SNOWFLAKE_ACCOUNT')
        self.user = os.getenv('SNOWFLAKE_USER')
        self.password = os.getenv('SNOWFLAKE_PASSWORD')
        self.connection = None

        if not all([self.account, self.user, self.password]):
            print("Warning: Snowflake credentials not found in environment variables")
            print("Using mock sentiment analysis instead")

    def connect(self):
        """Establish connection to Snowflake"""
        try:
            self.connection = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse='COMPUTE_WH',  # Default warehouse
                database='FINANCIAL_DATA',  # You'll need to create this
                schema='PUBLIC'
            )
            print("Successfully connected to Snowflake")
            return True
        except Exception as e:
            print(f"Failed to connect to Snowflake: {e}")
            return False

    def analyze_sentiment(self, texts):
        """Analyze sentiment using Snowflake Cortex"""
        if not texts:
            return []

        if not self.connection:
            if not self.connect():
                return self._mock_sentiment_analysis(texts)

        try:
            cursor = self.connection.cursor()
            results = []

            for text in texts:
                if not text or text.strip() == '':
                    results.append({
                        'text': text,
                        'sentiment_score': 0.0,
                        'classification': 'NEUTRAL',
                        'confidence': 0.5
                    })
                    continue

                # Clean text for SQL
                clean_text = text.replace("'", "''")[:500]  # Limit text length

                query = f"""
                SELECT 
                    SNOWFLAKE.CORTEX.SENTIMENT('{clean_text}') as sentiment_score,
                    SNOWFLAKE.CORTEX.CLASSIFY_TEXT(
                        '{clean_text}',
                        ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
                    ) as classification
                """

                cursor.execute(query)
                result = cursor.fetchone()

                if result:
                    sentiment_score = float(result[0]) if result[0] is not None else 0.0
                    classification = result[1] if result[1] is not None else 'NEUTRAL'

                    # Calculate confidence based on score magnitude
                    confidence = min(abs(sentiment_score) + 0.3, 1.0)

                    results.append({
                        'text': text,
                        'sentiment_score': sentiment_score,
                        'classification': classification,
                        'confidence': confidence
                    })
                else:
                    results.append({
                        'text': text,
                        'sentiment_score': 0.0,
                        'classification': 'NEUTRAL',
                        'confidence': 0.5
                    })

            cursor.close()
            return results

        except Exception as e:
            print(f"Error analyzing sentiment with Snowflake Cortex: {e}")
            return self._mock_sentiment_analysis(texts)

    def store_signals(self, signal_data):
        """Store generated signals in Snowflake"""
        if not self.connection:
            if not self.connect():
                print("Cannot store signals - no Snowflake connection")
                return False

        try:
            cursor = self.connection.cursor()

            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS TRADING_SIGNALS (
                timestamp TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
                symbol VARCHAR(10),
                signal VARCHAR(10),
                confidence FLOAT,
                rsi FLOAT,
                macd FLOAT,
                sentiment_score FLOAT,
                price FLOAT
            )
            """
            cursor.execute(create_table_query)

            # Insert signal data
            insert_query = """
            INSERT INTO TRADING_SIGNALS 
            (symbol, signal, confidence, rsi, macd, sentiment_score, price)
            VALUES (%(symbol)s, %(signal)s, %(confidence)s, %(rsi)s, %(macd)s, %(sentiment_score)s, %(price)s)
            """

            cursor.execute(insert_query, signal_data)
            cursor.close()

            print("Successfully stored signal data in Snowflake")
            return True

        except Exception as e:
            print(f"Error storing signals in Snowflake: {e}")
            return False

    def get_historical_signals(self, symbol, days=30):
        """Retrieve historical signals from Snowflake"""
        if not self.connection:
            if not self.connect():
                return self._mock_historical_signals(symbol, days)

        try:
            cursor = self.connection.cursor()

            query = """
            SELECT timestamp, symbol, signal, confidence, rsi, macd, sentiment_score, price
            FROM TRADING_SIGNALS 
            WHERE symbol = %(symbol)s 
            AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL '%(days)s DAYS'
            ORDER BY timestamp DESC
            LIMIT 100
            """

            cursor.execute(query, {'symbol': symbol, 'days': days})
            results = cursor.fetchall()
            cursor.close()

            columns = ['timestamp', 'symbol', 'signal', 'confidence', 'rsi', 'macd', 'sentiment_score', 'price']
            df = pd.DataFrame(results, columns=columns)

            return df

        except Exception as e:
            print(f"Error retrieving historical signals: {e}")
            return self._mock_historical_signals(symbol, days)

    def _mock_sentiment_analysis(self, texts):
        """Generate mock sentiment analysis when Snowflake is unavailable"""
        results = []

        for text in texts:
            # Simple keyword-based mock sentiment
            positive_words = ['strong', 'growth', 'profit', 'beat', 'exceed', 'positive', 'up', 'gain']
            negative_words = ['loss', 'down', 'fall', 'decline', 'weak', 'negative', 'drop', 'miss']

            if text:
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)

                if pos_count > neg_count:
                    sentiment_score = np.random.uniform(0.3, 0.8)
                    classification = 'POSITIVE'
                elif neg_count > pos_count:
                    sentiment_score = np.random.uniform(-0.8, -0.3)
                    classification = 'NEGATIVE'
                else:
                    sentiment_score = np.random.uniform(-0.2, 0.2)
                    classification = 'NEUTRAL'
            else:
                sentiment_score = 0.0
                classification = 'NEUTRAL'

            results.append({
                'text': text,
                'sentiment_score': sentiment_score,
                'classification': classification,
                'confidence': min(abs(sentiment_score) + 0.3, 1.0)
            })

        return results

    def _mock_historical_signals(self, symbol, days):
        """Generate mock historical signals"""
        from datetime import datetime, timedelta

        data = []
        for i in range(min(days, 30)):
            timestamp = datetime.now() - timedelta(days=i)
            signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.2, 0.5])

            data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'signal': signal,
                'confidence': np.random.uniform(0.3, 0.9),
                'rsi': np.random.uniform(20, 80),
                'macd': np.random.uniform(-2, 2),
                'sentiment_score': np.random.uniform(-0.5, 0.5),
                'price': np.random.uniform(100, 300)
            })

        return pd.DataFrame(data)

    def close_connection(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
