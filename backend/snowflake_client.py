from datetime import datetime, timedelta
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
    

    def get_cross_asset_correlation(self, symbols, window_days=30, correlation_threshold=0.6):
        """
        Generate cross-asset correlation radar using Snowflake Cortex.
        Returns a list of edges suitable for visualization (e.g., Plotly Network, D3.js).

        Args:
            symbols (list): List of asset tickers to compare.
            window_days (int): Time window for correlation calculation.
            correlation_threshold (float): Minimum correlation to display as edge.
        """
        if not self.connection:
            if not self.connect():
                print("⚠️ Using mock correlation data (Snowflake unavailable)")
                return self._mock_cross_asset_correlation(symbols)

        try:
            cursor = self.connection.cursor()

            # Prepare symbol list for SQL IN clause
            symbol_list = ",".join([f"'{s}'" for s in symbols])

            # Query last X days of price data
            query = f"""
            WITH price_data AS (
                SELECT symbol, timestamp::DATE AS dt, AVG(price) AS avg_price
                FROM TRADING_SIGNALS
                WHERE symbol IN ({symbol_list})
                AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL '{window_days} DAYS'
                GROUP BY symbol, dt
            ),
            pivoted AS (
                SELECT *
                FROM price_data
                PIVOT (AVG(avg_price) FOR symbol IN ({symbol_list}))
            )
            SELECT * FROM pivoted;
            """

            cursor.execute(query)
            df = pd.DataFrame(cursor.fetchall(), columns=[col[0] for col in cursor.description])

            # Drop date column if present and compute correlations
            df = df.select_dtypes(include=[np.number])
            corr_matrix = df.corr()

            # Build edges for network graph
            edges = []
            for i, sym1 in enumerate(corr_matrix.columns):
                for j, sym2 in enumerate(corr_matrix.columns):
                    if i < j:
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) >= correlation_threshold:
                            edges.append({
                                "source": sym1,
                                "target": sym2,
                                "correlation": round(float(corr_value), 3),
                                "relation": "POSITIVE" if corr_value > 0 else "NEGATIVE"
                            })

            cursor.close()
            print(f"✅ Generated {len(edges)} correlations above threshold {correlation_threshold}")
            return edges

        except Exception as e:
            print(f"Error generating cross-asset correlation: {e}")
            return self._mock_cross_asset_correlation(symbols)


    def _mock_cross_asset_correlation(self, symbols):
        """Generate mock correlation edges when Snowflake unavailable"""
        np.random.seed(42)
        edges = []
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                corr = np.random.uniform(-1, 1)
                if abs(corr) > 0.6:
                    edges.append({
                        "source": symbols[i],
                        "target": symbols[j],
                        "correlation": round(float(corr), 3),
                        "relation": "POSITIVE" if corr > 0 else "NEGATIVE"
                    })
        return edges
    


    def detect_anomalies(self, symbols, window_days=30, sentiment_z_thresh=2.5, vol_change_pct=0.4):
        """
        Detect anomalies for given symbols:
        - sentiment_spike: sentiment z-score exceeds sentiment_z_thresh
        - price_volume_divergence: price moves directionally while volume moves opposite beyond vol_change_pct
        Returns list of anomalies: {symbol, date, type, details}
        """
        if not self.connection:
            if not self.connect():
                return self._mock_anomalies(symbols)

        try:
            cursor = self.connection.cursor()
            symbol_list = ",".join([f"'{s}'" for s in symbols])

            query = f"""
            SELECT symbol, DATE_TRUNC('DAY', timestamp) AS dt,
                AVG(price) AS avg_price,
                AVG(volume) AS avg_volume,
                AVG(sentiment_score) AS avg_sentiment
            FROM TRADING_SIGNALS
            WHERE symbol IN ({symbol_list})
            AND timestamp >= CURRENT_TIMESTAMP() - INTERVAL '{window_days} DAYS'
            GROUP BY symbol, dt
            ORDER BY symbol, dt;
            """

            cursor.execute(query)
            rows = cursor.fetchall()
            cols = [c[0] for c in cursor.description]
            df = pd.DataFrame(rows, columns=cols)

            if df.empty:
                cursor.close()
                return []

            anomalies = []
            for sym, g in df.groupby('SYMBOL' if 'SYMBOL' in df.columns else 'symbol'):
                # normalize column names
                g = g.rename(columns={c: c.lower() for c in g.columns})
                g = g.sort_values('dt')
                # sentiment spike detection via z-score
                if 'avg_sentiment' in g.columns:
                    s = g['avg_sentiment'].astype(float)
                    mean = s.mean()
                    std = s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0
                    z = (s - mean) / std
                    spikes = g.loc[z.abs() >= sentiment_z_thresh]
                    for _, row in spikes.iterrows():
                        anomalies.append({
                            "symbol": sym,
                            "date": str(row['dt']),
                            "type": "sentiment_spike",
                            "severity": float(round(abs((row['avg_sentiment'] - mean) / std), 3)),
                            "detail": f"sentiment={row['avg_sentiment']:.3f}, z={((row['avg_sentiment']-mean)/std):.2f}"
                        })

                # price-volume divergence
                if {'avg_price', 'avg_volume'}.issubset(set(g.columns)):
                    g['price_pct'] = g['avg_price'].pct_change().fillna(0)
                    g['vol_pct'] = g['avg_volume'].pct_change().fillna(0)
                    # price up & volume down OR price down & volume up
                    divergences = g[((g['price_pct'] > 0.02) & (g['vol_pct'] < -vol_change_pct)) |
                                    ((g['price_pct'] < -0.02) & (g['vol_pct'] > vol_change_pct))]
                    for _, row in divergences.iterrows():
                        anomalies.append({
                            "symbol": sym,
                            "date": str(row['dt']),
                            "type": "price_volume_divergence",
                            "severity": float(round(max(abs(row['price_pct']), abs(row['vol_pct'])), 3)),
                            "detail": f"price_pct={row['price_pct']:.3f}, vol_pct={row['vol_pct']:.3f}"
                        })

            cursor.close()
            print(f"✅ Detected {len(anomalies)} anomalies for window {window_days}d")
            return anomalies

        except Exception as e:
            print(f"Error in detect_anomalies: {e}")
            return self._mock_anomalies(symbols)


    def _mock_anomalies(self, symbols):
        """Return synthetic anomalies when Snowflake not available"""
        np.random.seed(123)
        anomalies = []
        for sym in symbols:
            if np.random.rand() > 0.7:
                anomalies.append({
                    "symbol": sym,
                    "date": (datetime.now() - timedelta(days=int(np.random.rand()*5))).strftime("%Y-%m-%d"),
                    "type": np.random.choice(["sentiment_spike", "price_volume_divergence"]),
                    "severity": float(round(np.random.uniform(0.5, 3.0), 3)),
                    "detail": "mock anomaly"
                })
        return anomalies




    def close_connection(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            self.connection = None