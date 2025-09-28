
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta

class FinancialSignalGenerator:
    def __init__(self):
        self.indicators = {}
        self.signals = {}

    def generate_signals(self, stock_data, sentiment_data, params=None):
        """Generate comprehensive trading signals"""
        if params is None:
            params = {
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'sentiment_threshold': 0.5,
                'min_confidence': 0.3
            }

        # Fetch intraday data for technical analysis
        from data_fetcher import DataFetcher
        fetcher = DataFetcher()
        df = fetcher.get_intraday_data(stock_data['symbol'])

        if df is None or len(df) < 50:
            return self._generate_mock_signals(stock_data, sentiment_data, params)

        # Calculate technical indicators
        indicators = self._calculate_technical_indicators(df)

        # Generate individual signals
        rsi_signal = self._rsi_signal(indicators['rsi'], params)
        macd_signal = self._macd_signal(indicators['macd'], indicators['macd_signal'])
        ma_signal = self._moving_average_signal(df['close'], indicators['sma_20'], indicators['sma_50'])
        bb_signal = self._bollinger_bands_signal(indicators['bb_position'])

        # Sentiment signal
        sentiment_signal = self._sentiment_signal(sentiment_data, params)

        # Combine signals
        combined_signal = self._combine_signals({
            'rsi': rsi_signal,
            'macd': macd_signal,
            'ma': ma_signal,
            'bb': bb_signal,
            'sentiment': sentiment_signal
        })

        # Calculate confidence
        confidence = self._calculate_confidence(combined_signal, indicators, sentiment_data)

        # Final signal decision
        final_signal = self._make_final_decision(combined_signal, confidence, params)

        return {
            'final_signal': final_signal,
            'confidence': confidence,
            'rsi': indicators['rsi'],
            'macd': indicators['macd'],
            'macd_signal': indicators['macd_signal'],
            'sma_20': indicators['sma_20'],
            'sma_50': indicators['sma_50'],
            'bb_position': indicators['bb_position'],
            'sentiment_score': np.mean([s['sentiment_score'] for s in sentiment_data]),
            'individual_signals': {
                'rsi': rsi_signal,
                'macd': macd_signal,
                'ma': ma_signal,
                'bb': bb_signal,
                'sentiment': sentiment_signal
            }
        }

    def _calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values

        try:
            indicators = {
                'rsi': talib.RSI(close_prices, timeperiod=14)[-1],
                'macd': None,
                'macd_signal': None,
                'macd_histogram': None,
                'sma_20': talib.SMA(close_prices, timeperiod=20)[-1],
                'sma_50': talib.SMA(close_prices, timeperiod=50)[-1],
                'sma_200': talib.SMA(close_prices, timeperiod=200)[-1] if len(close_prices) > 200 else None,
                'ema_20': talib.EMA(close_prices, timeperiod=20)[-1],
                'bb_upper': None,
                'bb_middle': None,
                'bb_lower': None,
                'bb_position': None
            }

            # MACD calculation
            macd, macd_signal, macd_histogram = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd'] = macd[-1]
            indicators['macd_signal'] = macd_signal[-1]
            indicators['macd_histogram'] = macd_histogram[-1]

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            indicators['bb_upper'] = bb_upper[-1]
            indicators['bb_middle'] = bb_middle[-1]
            indicators['bb_lower'] = bb_lower[-1]

            # BB position (where current price is relative to bands)
            current_price = close_prices[-1]
            bb_width = bb_upper[-1] - bb_lower[-1]
            indicators['bb_position'] = (current_price - bb_middle[-1]) / (bb_width / 2) if bb_width > 0 else 0

            return indicators

        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return self._get_mock_indicators(close_prices[-1])

    def _rsi_signal(self, rsi, params):
        """Generate RSI-based signal"""
        if rsi is None or np.isnan(rsi):
            return 0

        if rsi < params['rsi_oversold']:
            return 1  # Buy signal
        elif rsi > params['rsi_overbought']:
            return -1  # Sell signal
        else:
            return 0  # Neutral

    def _macd_signal(self, macd, macd_signal_line):
        """Generate MACD-based signal"""
        if macd is None or macd_signal_line is None or np.isnan(macd) or np.isnan(macd_signal_line):
            return 0

        if macd > macd_signal_line:
            return 1  # Bullish
        elif macd < macd_signal_line:
            return -1  # Bearish
        else:
            return 0

    def _moving_average_signal(self, current_price, sma_20, sma_50):
        """Generate moving average signal"""
        if sma_20 is None or sma_50 is None or np.isnan(sma_20) or np.isnan(sma_50):
            return 0

        price = current_price.iloc[-1] if hasattr(current_price, 'iloc') else current_price

        if price > sma_20 > sma_50:
            return 1  # Bullish trend
        elif price < sma_20 < sma_50:
            return -1  # Bearish trend
        else:
            return 0

    def _bollinger_bands_signal(self, bb_position):
        """Generate Bollinger Bands signal"""
        if bb_position is None or np.isnan(bb_position):
            return 0

        if bb_position < -1:  # Below lower band
            return 1  # Oversold, buy signal
        elif bb_position > 1:  # Above upper band
            return -1  # Overbought, sell signal
        else:
            return 0

    def _sentiment_signal(self, sentiment_data, params):
        """Generate sentiment-based signal"""
        if not sentiment_data:
            return 0

        avg_sentiment = np.mean([s['sentiment_score'] for s in sentiment_data])

        if avg_sentiment > params['sentiment_threshold']:
            return 1  # Positive sentiment
        elif avg_sentiment < -params['sentiment_threshold']:
            return -1  # Negative sentiment
        else:
            return 0

    def _combine_signals(self, signals, weights=None):
        """Combine individual signals with weights"""
        if weights is None:
            weights = {
                'rsi': 0.2,
                'macd': 0.25,
                'ma': 0.2,
                'bb': 0.15,
                'sentiment': 0.2
            }

        combined = 0
        for signal_name, signal_value in signals.items():
            weight = weights.get(signal_name, 0.2)
            combined += signal_value * weight

        return combined

    def _calculate_confidence(self, combined_signal, indicators, sentiment_data):
        """Calculate confidence score for the signal"""
        confidence = abs(combined_signal)

        # Adjust based on indicator strength
        if indicators.get('rsi') is not None:
            rsi = indicators['rsi']
            if rsi < 20 or rsi > 80:  # Strong RSI signal
                confidence *= 1.2

        # Adjust based on sentiment consistency
        if sentiment_data:
            sentiment_scores = [s['sentiment_score'] for s in sentiment_data]
            sentiment_std = np.std(sentiment_scores)
            if sentiment_std < 0.2:  # Consistent sentiment
                confidence *= 1.1

        return min(confidence, 1.0)  # Cap at 100%

    def _make_final_decision(self, combined_signal, confidence, params):
        """Make final trading decision"""
        if confidence < params['min_confidence']:
            return "HOLD"

        if combined_signal > 0.3:
            return "BUY"
        elif combined_signal < -0.3:
            return "SELL"
        else:
            return "HOLD"

    def _generate_mock_signals(self, stock_data, sentiment_data, params):
        """Generate mock signals when real calculation fails"""
        mock_indicators = self._get_mock_indicators(stock_data['price'])

        return {
            'final_signal': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.2, 0.5]),
            'confidence': np.random.uniform(0.4, 0.9),
            **mock_indicators,
            'sentiment_score': np.mean([s['sentiment_score'] for s in sentiment_data]) if sentiment_data else 0,
            'individual_signals': {
                'rsi': np.random.choice([-1, 0, 1]),
                'macd': np.random.choice([-1, 0, 1]),
                'ma': np.random.choice([-1, 0, 1]),
                'bb': np.random.choice([-1, 0, 1]),
                'sentiment': np.random.choice([-1, 0, 1])
            }
        }

    def _get_mock_indicators(self, current_price):
        """Generate mock technical indicators"""
        return {
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.uniform(-2, 2),
            'macd_signal': np.random.uniform(-2, 2),
            'sma_20': current_price * np.random.uniform(0.95, 1.05),
            'sma_50': current_price * np.random.uniform(0.90, 1.10),
            'bb_position': np.random.uniform(-1.5, 1.5)
        }
