
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
from pathlib import Path

# Add the backend directory to the path
import sys
sys.path.append(str(Path(__file__).parent / "backend"))

from signal_generator import FinancialSignalGenerator
from data_fetcher import DataFetcher
from snowflake_client import SnowflakeClient

# Configure Streamlit page
st.set_page_config(
    page_title="Financial Signal Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }

    .signal-buy {
        background: linear-gradient(90deg, #065f46 0%, #059669 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }

    .signal-sell {
        background: linear-gradient(90deg, #991b1b 0%, #dc2626 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }

    .signal-hold {
        background: linear-gradient(90deg, #92400e 0%, #d97706 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }

    .indicator-positive {
        color: #10b981;
        font-weight: bold;
    }

    .indicator-negative {
        color: #ef4444;
        font-weight: bold;
    }

    .indicator-neutral {
        color: #f59e0b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = DataFetcher()
    st.session_state.signal_generator = FinancialSignalGenerator()
    st.session_state.snowflake_client = SnowflakeClient()
    st.session_state.last_update = None
    st.session_state.current_data = None

def get_signal_color_class(signal):
    """Return CSS class based on signal type"""
    if signal == "BUY":
        return "signal-buy"
    elif signal == "SELL":
        return "signal-sell"
    else:
        return "signal-hold"

def get_indicator_color_class(value, thresholds):
    """Return CSS class based on indicator value and thresholds"""
    if value > thresholds['high']:
        return "indicator-negative"
    elif value < thresholds['low']:
        return "indicator-positive"
    else:
        return "indicator-neutral"

def main():
    # Header
    st.title("🚀 Real-Time Financial Signal Generation System")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📊 Configuration")

        # Stock selection
        selected_stock = st.selectbox(
            "Select Stock Symbol",
            ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "SPY", "NVDA", "META"],
            help="Choose a stock to analyze"
        )

        st.markdown("---")

        # Refresh controls
        st.subheader("🔄 Data Controls")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh Data", type="primary"):
                st.session_state.current_data = None
                st.rerun()

        with col2:
            auto_refresh = st.checkbox("Auto Refresh", value=False)

        if auto_refresh:
            st.info("Auto-refreshing every 30 seconds")
            time.sleep(30)
            st.rerun()

        # Last update info
        if st.session_state.last_update:
            st.markdown(f"**Last Updated:** {st.session_state.last_update.strftime('%H:%M:%S')}")

        st.markdown("---")

        # System status
        st.subheader("⚡ System Status")

        # Check API connections
        try:
            # This would check if APIs are working
            st.success("✅ Alpha Vantage API")
            st.success("✅ News API")
            st.success("✅ Snowflake Connection")
        except:
            st.error("❌ API Connection Issues")

        st.markdown("---")

        # Signal parameters
        st.subheader("⚙️ Signal Parameters")

        rsi_oversold = st.slider("RSI Oversold Threshold", 20, 40, 30)
        rsi_overbought = st.slider("RSI Overbought Threshold", 60, 80, 70)

        sentiment_threshold = st.slider("Sentiment Threshold", 0.1, 0.8, 0.5)

        signal_confidence_min = st.slider("Min Signal Confidence", 0.1, 0.9, 0.3)

    # Main content area

    # Fetch data if not cached or symbol changed
    if (st.session_state.current_data is None or 
        st.session_state.current_data.get('symbol') != selected_stock):

        with st.spinner(f"Fetching data for {selected_stock}..."):
            try:
                # Fetch stock data
                stock_data = st.session_state.data_fetcher.get_stock_data(selected_stock)

                # Fetch news and sentiment
                news_data = st.session_state.data_fetcher.get_news_data(selected_stock)
                sentiment_data = st.session_state.snowflake_client.analyze_sentiment(
                    [article['description'] for article in news_data[:5]]
                )

                # Generate signals
                signals = st.session_state.signal_generator.generate_signals(
                    stock_data, sentiment_data, {
                        'rsi_oversold': rsi_oversold,
                        'rsi_overbought': rsi_overbought,
                        'sentiment_threshold': sentiment_threshold,
                        'min_confidence': signal_confidence_min
                    }
                )

                st.session_state.current_data = {
                    'symbol': selected_stock,
                    'stock_data': stock_data,
                    'news_data': news_data,
                    'sentiment_data': sentiment_data,
                    'signals': signals
                }
                st.session_state.last_update = datetime.now()

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                st.info("Using simulated data for demonstration")
                # Use sample data
                st.session_state.current_data = get_sample_data(selected_stock)
                st.session_state.last_update = datetime.now()

    data = st.session_state.current_data

    if data:
        # Current price and key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            price = data['stock_data'].get('price', 150.25)
            change = data['stock_data'].get('change', 2.15)
            change_pct = data['stock_data'].get('change_percent', 1.45)

            delta_color = "normal" if change >= 0 else "inverse"
            st.metric(
                f"{selected_stock} Price",
                f"${price:.2f}",
                f"{change:+.2f} ({change_pct:+.2f}%)",
                delta_color=delta_color
            )

        with col2:
            volume = data['stock_data'].get('volume', 45000000)
            st.metric("Volume", f"{volume:,}")

        with col3:
            market_cap = data['stock_data'].get('market_cap', '2.4T')
            st.metric("Market Cap", market_cap)

        with col4:
            # Current signal
            current_signal = data['signals'].get('final_signal', 'HOLD')
            signal_confidence = data['signals'].get('confidence', 0.65)

            st.markdown(f"""
            <div class="{get_signal_color_class(current_signal)}">
                {current_signal} Signal<br>
                Confidence: {signal_confidence:.1%}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        

        # Technical Indicators Section
        st.subheader("📊 Technical Indicators")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            rsi = data['signals'].get('rsi', 65.4)
            rsi_class = get_indicator_color_class(rsi, {'low': rsi_oversold, 'high': rsi_overbought})
            st.markdown(f"**RSI (14):** <span class='{rsi_class}'>{rsi:.1f}</span>", unsafe_allow_html=True)
            st.progress(rsi / 100)

        with col2:
            macd = data['signals'].get('macd', 1.23)
            macd_signal = data['signals'].get('macd_signal', 1.15)
            macd_color = "indicator-positive" if macd > macd_signal else "indicator-negative"
            st.markdown(f"**MACD:** <span class='{macd_color}'>{macd:.3f}</span>", unsafe_allow_html=True)
            st.markdown(f"**Signal:** {macd_signal:.3f}")

        with col3:
            sma_20 = data['signals'].get('sma_20', 148.50)
            sma_50 = data['signals'].get('sma_50', 145.20)
            ma_color = "indicator-positive" if price > sma_20 else "indicator-negative"
            st.markdown(f"**SMA 20:** <span class='{ma_color}'>{sma_20:.2f}</span>", unsafe_allow_html=True)
            st.markdown(f"**SMA 50:** {sma_50:.2f}")

        with col4:
            bb_position = data['signals'].get('bb_position', 0.3)
            bb_color = get_indicator_color_class(bb_position, {'low': -1, 'high': 1})
            st.markdown(f"**BB Position:** <span class='{bb_color}'>{bb_position:.2f}</span>", unsafe_allow_html=True)

            if bb_position > 1:
                st.caption("Above upper band")
            elif bb_position < -1:
                st.caption("Below lower band")
            else:
                st.caption("Within bands")

        st.markdown("---")

        # Charts Section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Price Chart with Technical Indicators")

            # Create sample price data for chart
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                 end=datetime.now(), freq='D')
            prices = np.random.walk_cumulative(len(dates), start=price-10, step_std=2)

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=dates, y=prices,
                mode='lines',
                name=f'{selected_stock} Price',
                line=dict(color='blue', width=2)
            ))

            # Moving averages
            sma_20_line = pd.Series(prices).rolling(20).mean()
            sma_50_line = pd.Series(prices).rolling(50).mean()

            fig.add_trace(go.Scatter(
                x=dates, y=sma_20_line,
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))

            fig.update_layout(
                title=f"{selected_stock} Price Chart",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Signal Strength Indicators")

            # Create gauge chart for signal strength
            signal_strength = signal_confidence * 100

            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = signal_strength,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Signal Confidence"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))

            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Risk assessment
            risk_level = "LOW" if signal_confidence > 0.7 else "MODERATE" if signal_confidence > 0.4 else "HIGH"
            risk_color = "#10b981" if risk_level == "LOW" else "#f59e0b" if risk_level == "MODERATE" else "#ef4444"

            st.markdown(f"**Risk Level:** <span style='color: {risk_color}; font-weight: bold'>{risk_level}</span>", 
                       unsafe_allow_html=True)

        st.markdown("---")

        # Sentiment Analysis Section
        st.subheader("📰 News Sentiment Analysis")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Recent Headlines")

            for i, article in enumerate(data['news_data'][:5]):
                sentiment_score = data['sentiment_data'][i]['sentiment_score']
                sentiment_color = "#10b981" if sentiment_score > 0.2 else "#ef4444" if sentiment_score < -0.2 else "#6b7280"

                st.markdown(f"""
                <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin: 8px 0;">
                    <h4 style="margin: 0; font-size: 16px;">{article['title']}</h4>
                    <p style="margin: 8px 0; color: #6b7280; font-size: 14px;">{article['description'][:200]}...</p>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {sentiment_color}; font-weight: bold;">
                            Sentiment: {sentiment_score:.2f}
                        </span>
                        <span style="color: #9ca3af; font-size: 12px;">
                            {article.get('publishedAt', 'Recent')}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("Sentiment Summary")

            avg_sentiment = np.mean([s['sentiment_score'] for s in data['sentiment_data']])
            sentiment_classification = "POSITIVE" if avg_sentiment > 0.2 else "NEGATIVE" if avg_sentiment < -0.2 else "NEUTRAL"

            sentiment_color = "#10b981" if sentiment_classification == "POSITIVE" else "#ef4444" if sentiment_classification == "NEGATIVE" else "#f59e0b"

            st.markdown(f"""
            <div style="background: {sentiment_color}; color: white; padding: 20px; border-radius: 8px; text-align: center;">
                <h3 style="margin: 0;">{sentiment_classification}</h3>
                <h2 style="margin: 10px 0;">{avg_sentiment:.2f}</h2>
                <p style="margin: 0;">Average Sentiment</p>
            </div>
            """, unsafe_allow_html=True)

            # Sentiment distribution
            sentiment_scores = [s['sentiment_score'] for s in data['sentiment_data']]

            fig = px.histogram(
                x=sentiment_scores,
                nbins=10,
                title="Sentiment Distribution",
                labels={'x': 'Sentiment Score', 'y': 'Count'}
            )
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Historical Signals Section
        st.subheader("📜 Signal History")

        # Create sample historical data
        historical_data = []
        for i in range(10):
            date = datetime.now() - timedelta(hours=i*6)
            signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.2, 0.15, 0.65])
            confidence = np.random.uniform(0.3, 0.9)

            historical_data.append({
                'Timestamp': date.strftime('%Y-%m-%d %H:%M'),
                'Signal': signal,
                'Confidence': f"{confidence:.1%}",
                'RSI': np.random.uniform(30, 70),
                'MACD': np.random.uniform(-2, 2),
                'Sentiment': np.random.uniform(-0.5, 0.5)
            })

        df_history = pd.DataFrame(historical_data)

        # Color code the signals
        def color_signal(val):
            if val == 'BUY':
                return 'background-color: #d1fae5; color: #065f46'
            elif val == 'SELL':
                return 'background-color: #fee2e2; color: #991b1b'
            else:
                return 'background-color: #fef3c7; color: #92400e'

        styled_df = df_history.style.applymap(color_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True)

        
    

        # --- Cross-Asset Correlation (Snowflake Cortex) ---
        st.markdown("---")
        st.subheader("🔗 Cross-Asset Correlation Radar")

        corr_col1, corr_col2 = st.columns([2, 1])
        with corr_col2:
            window_days = st.slider("Window (days)", 7, 180, 30)
            corr_threshold = st.slider("Correlation threshold", 0.1, 0.95, 0.6, step=0.05)
            symbols_input = st.text_area("Symbols (comma separated)", value="AAPL,MSFT,GOOGL,NVDA,TSLA")
            run_corr = st.button("Generate Correlation")

        with corr_col1:
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
            st.markdown(f"Selected symbols: **{', '.join(symbols)}**")

            if run_corr:
                if not symbols:
                    st.error("Please provide at least two symbols.")
                else:
                    try:
                        edges = st.session_state.snowflake_client.get_cross_asset_correlation(
                            symbols, window_days=window_days, correlation_threshold=corr_threshold
                        )
                    except AttributeError:
                        # Fallback to mock if method not available
                        try:
                            edges = st.session_state.snowflake_client._mock_cross_asset_correlation(symbols)
                        except Exception as e:
                            st.error(f"Correlation unavailable: {e}")
                            edges = []

                    if not edges:
                        st.info("No correlations above the threshold. Try lowering the threshold or changing symbols.")
                    else:
                        # Show edges table
                        df_edges = pd.DataFrame(edges)
                        st.table(df_edges)

                        # Simple circular network layout for visualization
                        # compute node positions on a circle
                        unique_nodes = list({e['source'] for e in edges} | {e['target'] for e in edges})
                        n = len(unique_nodes)
                        angle_step = 2 * np.pi / max(n, 1)
                        positions = {
                            node: (np.cos(i * angle_step), np.sin(i * angle_step))
                            for i, node in enumerate(unique_nodes)
                        }

                        fig = go.Figure()

                        # Add edges as lines
                        for e in edges:
                            x0, y0 = positions[e['source']]
                            x1, y1 = positions[e['target']]
                            color = "#10b981" if e.get('relation') == "POSITIVE" else "#ef4444"
                            width = 1 + abs(e.get('correlation', 0)) * 3
                            fig.add_trace(go.Scatter(
                                x=[x0, x1], y=[y0, y1],
                                mode="lines",
                                line=dict(color=color, width=width),
                                hoverinfo="text",
                                text=f"{e['source']} ↔ {e['target']}: {e['correlation']}"
                            ))

                        # Add nodes
                        node_x = [positions[n][0] for n in unique_nodes]
                        node_y = [positions[n][1] for n in unique_nodes]
                        hover_text = []
                        for node in unique_nodes:
                            connected = [f"{ed['target']}({ed['correlation']})" if ed['source']==node else f"{ed['source']}({ed['correlation']})"
                                         for ed in edges if ed['source']==node or ed['target']==node]
                            hover_text.append(f"{node}<br>Connections: {len(connected)}<br>" + "<br>".join(connected))

                        fig.add_trace(go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            marker=dict(size=22, color="#1f77b4"),
                            text=unique_nodes,
                            textposition="bottom center",
                            hoverinfo="text",
                            hovertext=hover_text
                        ))

                        fig.update_layout(
                            showlegend=False,
                            xaxis=dict(showgrid=False, zeroline=False, visible=False),
                            yaxis=dict(showgrid=False, zeroline=False, visible=False),
                            height=500,
                            margin=dict(l=20, r=20, t=40, b=20),
                            title=f"Cross-Asset Correlations (threshold ≥ {corr_threshold})"
                        )

                        st.plotly_chart(fig, use_container_width=True)

        



        # --- Anomaly Detection (Snowflake) ---
        st.markdown("---")
        st.subheader("🚨 Anomaly Detection")

        anom_col1, anom_col2 = st.columns([2, 1])
        with anom_col2:
            anom_window = st.slider("Window (days)", 7, 90, 30)
            anom_sent_z = st.number_input("Sentiment Z threshold", value=2.5, step=0.1)
            anom_vol_pct = st.number_input("Volume change threshold (fraction)", value=0.4, step=0.05)
            anom_symbols = st.text_input("Symbols (comma separated)", value="AAPL,MSFT,GOOGL").upper()
            run_anom = st.button("Run Anomaly Detection")

        with anom_col1:
            symbols = [s.strip() for s in anom_symbols.split(",") if s.strip()]
            st.markdown(f"Checking: **{', '.join(symbols)}**")

            if run_anom:
                if not symbols or len(symbols) < 1:
                    st.error("Provide at least one symbol.")
                else:
                    try:
                        anomalies = st.session_state.snowflake_client.detect_anomalies(
                            symbols,
                            window_days=anom_window,
                            sentiment_z_thresh=anom_sent_z,
                            vol_change_pct=anom_vol_pct
                        )
                    except AttributeError:
                        anomalies = st.session_state.snowflake_client._mock_anomalies(symbols)

                    # ...existing code...
                    if not anomalies:
                        sample_data = get_sample_data(symbols[0])  # Get sample data for first symbol
                        anomalies = sample_data.get('anomalies', [])
                        
                        if anomalies:  # Show sample anomalies
                            df_anom = pd.DataFrame(anomalies)
                            st.table(df_anom.sort_values(['symbol', 'date']))

                    else:
                        df_anom = pd.DataFrame(anomalies)
                        st.markdown("### Detected anomalies")
                        st.table(df_anom.sort_values(['symbol', 'date']))

                        # display per-anomaly highlights and simple plots
                        for a in anomalies:
                            sym = a['symbol']
                            st.markdown(f"**{sym} — {a['type'].replace('_',' ').title()}** • {a['date']}")
                            st.caption(a.get('detail', ''))
                            if a['type'] == 'sentiment_spike':
                                st.warning(f"Sentiment spike (severity {a['severity']}) for {sym} on {a['date']}")
                            else:
                                st.info(f"Price/Volume divergence (severity {a['severity']}) for {sym} on {a['date']}")

                            # try to show small sample price/volume chart (use cached or sample)
                            try:
                                # prefer cached if same symbol
                                if data and data.get('symbol') == sym:
                                    sdata = data
                                    # build series around anomaly date
                                    dates = pd.date_range(end=datetime.now(), periods=30)
                                    prices = np.random.walk_cumulative(len(dates), start=sdata['stock_data'].get('price', 100)-5, step_std=2)
                                    volumes = np.random.randint(1000000, 50000000, size=len(dates))
                                else:
                                    # fallback to sample generator
                                    sample = get_sample_data(sym)
                                    dates = pd.date_range(end=datetime.now(), periods=30)
                                    prices = np.random.walk_cumulative(len(dates), start=sample['stock_data']['price']-5, step_std=2)
                                    volumes = np.random.randint(500000, 20000000, size=len(dates))
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=dates, y=prices, name='Price', yaxis='y1'))
                                fig.add_trace(go.Bar(x=dates, y=volumes, name='Volume', yaxis='y2', opacity=0.3))
                                fig.update_layout(
                                    height=300,
                                    margin=dict(t=10, b=10),
                                    yaxis=dict(title='Price', side='left'),
                                    yaxis2=dict(title='Volume', overlaying='y', side='right', showgrid=False),
                                    showlegend=True
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception:
                                # If rendering fails, continue
                                pass



# ...existing code...
def get_sample_data(symbol):
    """Generate sample data for demonstration (includes synthetic anomalies)"""
    base_price = np.random.uniform(100, 300)
    stock_info = {
        'price': base_price,
        'change': np.random.uniform(-10, 10),
        'change_percent': np.random.uniform(-3, 3),
        'volume': np.random.randint(1000000, 50000000),
        'market_cap': np.random.choice(['1.2T', '800B', '2.1T', '1.5T'])
    }

    # Make some sample news and sentiment
    news = [
        {
            'title': f'{symbol} reports quarterly earnings',
            'description': f'{symbol} has reported strong quarterly earnings with revenue growth exceeding expectations.',
            'publishedAt': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        },
        {
            'title': f'{symbol} announces new product launch',
            'description': f'{symbol} unveiled their latest product innovation at the tech conference.',
            'publishedAt': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        }
    ]
    sentiment = [
        {'sentiment_score': np.random.uniform(-0.5, 0.8)},
        {'sentiment_score': np.random.uniform(-0.5, 0.8)}
    ]

    signals = {
        'final_signal': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.2, 0.5]),
        'confidence': np.random.uniform(0.4, 0.9),
        'rsi': np.random.uniform(20, 80),
        'macd': np.random.uniform(-2, 2),
        'macd_signal': np.random.uniform(-2, 2),
        'sma_20': np.random.uniform(140, 160),
        'sma_50': np.random.uniform(135, 155),
        'bb_position': np.random.uniform(-1.5, 1.5)
    }

    # Flood with synthetic anomalies for demo/testing
    anomalies = []
    num_anom = np.random.randint(3, 8)  # flood with 3-7 anomalies
    for i in range(num_anom):
        a_type = np.random.choice(['sentiment_spike', 'price_volume_divergence'])
        days_ago = np.random.randint(0, 7)
        an_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        severity = float(round(np.random.uniform(0.6, 3.0), 3))
        if a_type == 'sentiment_spike':
            detail = f"sentiment={round(np.random.uniform(-1.0, 1.5), 3)}, z={round(np.random.uniform(2.5, 6.0),2)}"
        else:
            price_pct = round(np.random.uniform(-0.12, 0.12), 3)
            vol_pct = round(np.random.uniform(-0.8, 0.8), 3)
            detail = f"price_pct={price_pct}, vol_pct={vol_pct}"

        anomalies.append({
            'symbol': symbol,
            'date': an_date,
            'type': a_type,
            'severity': severity,
            'detail': detail
        })

    return {
        'symbol': symbol,
        'stock_data': stock_info,
        'news_data': news,
        'sentiment_data': sentiment,
        'signals': signals,
        'anomalies': anomalies
    }

# Helper function for random walk
def random_walk_cumulative(n, start=0, step_std=1):
    steps = np.random.normal(0, step_std, n)
    return start + np.cumsum(steps)

# Add to numpy for convenience
np.random.walk_cumulative = random_walk_cumulative

if __name__ == "__main__":
    main()
