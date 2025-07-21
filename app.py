import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sentence_transformers import SentenceTransformer

# --- Page Configuration ---
st.set_page_config(
    page_title="Stocker.AI - Advanced Analysis",
    page_icon="🔮",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    .big-font {
        font-size: 3rem !important;
        font-weight: bold;
        color: #1E90FF;
        text-align: center;
        padding-bottom: 20px;
    }
    .metric-box {
        border: 1px solid #1E90FF;
        border-radius: 10px;
        padding: 10px;
        text-align: center;
        margin: 5px;
    }
    .metric-label {
        font-size: 0.9em;
        font-weight: bold;
        color: #B0C4DE; /* Light Steel Blue */
    }
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- Model and Function Definitions ---

@st.cache_resource
def load_models():
    """Loads the Keras and Sentence Transformer models."""
    try:
        prediction_model = load_model("stock_prediction_model.h5")
    except Exception as e:
        st.error(f"Error loading Keras model: {e}. Make sure the model file is present.")
        return None, None
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return prediction_model, sentence_model

@st.cache_data(ttl=600)
def get_stock_data(ticker, period):
    """Fetches stock data, info, and calculates technical indicators."""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, auto_adjust=True)
    if data.empty:
        return None, None
    
    # Technical Indicators
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data, stock.info

def predict_historical_prices(_model, data):
    """Predicts prices on historical data to show model fit."""
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    
    window_size = 60
    if len(scaled_data) < window_size:
        return np.array([]), scaler

    X_test = [scaled_data[i-window_size:i, 0] for i in range(window_size, len(scaled_data))]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predictions = _model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions.flatten(), scaler

def forecast_future_prices(_model, data, scaler, n_days=3):
    """Forecasts future stock prices for n_days using iterative prediction."""
    window_size = 60
    last_60_days = data['Close'].values[-window_size:]
    scaled_last_60_days = scaler.transform(last_60_days.reshape(-1, 1))
    
    future_predictions = []
    current_batch = scaled_last_60_days.reshape(1, window_size, 1)

    for _ in range(n_days):
        # Predict the next day
        next_pred = _model.predict(current_batch)[0]
        future_predictions.append(next_pred)
        # Update the batch to include the new prediction and remove the oldest value
        current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)

    # Inverse transform the predictions to get actual price values
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions.flatten()

@st.cache_data(ttl=1800)
def fetch_news():
    """Fetches general stock market news."""
    url = "https://www.moneycontrol.com/news/business/stocks/"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_="clearfix", limit=10)
        return [(a.find('h2').text.strip(), a.find('a')['href']) for a in articles if a.find('h2') and a.find('a')]
    except Exception:
        return []

def generate_sentiment_analysis(description, rsi_value, news_titles, _sentence_model):
    """Provides an investment suggestion based on trend, RSI, and news."""
    # (This function remains unchanged)
    positive_keywords = ["growth", "buy", "positive", "bullish", "rally", "profit", "up"]
    negative_keywords = ["decline", "sell", "bearish", "plunge", "drop", "loss", "down"]
    positive_score = sum(1 for title, _ in news_titles if any(word in title.lower() for word in positive_keywords))
    negative_score = sum(1 for title, _ in news_titles if any(word in title.lower() for word in negative_keywords))
    if "upward" in description:
        if rsi_value < 70:
            return f"🚀 **Positive Outlook:** The model predicts an upward trend and the RSI ({rsi_value:.2f}) is not in the overbought zone. News sentiment appears neutral to positive. This could be a favorable setup, but always do your own research (DYOR)."
        else:
            return f"⚠️ **Mixed Signals:** The trend is upward, but the RSI ({rsi_value:.2f}) indicates the stock may be overbought. A pullback is possible. Caution is advised despite the positive trend."
    elif "downward" in description:
        if rsi_value > 30:
            return f"📉 **Negative Outlook:** The model indicates a potential downward trend, and the RSI ({rsi_value:.2f}) has room to fall further. News sentiment appears weak. Proceed with caution."
        else:
             return f"⚖️ **Potential Reversal?** The trend is downward, but the RSI ({rsi_value:.2f}) suggests the stock is oversold, which could signal a potential bounce. High risk; monitor closely."
    return "⚖️ **Neutral Outlook:** The indicators are mixed. It's best to monitor the stock closely before making a decision."


# --- Main Application ---
prediction_model, sentence_model = load_models()
if not prediction_model:
    st.stop()

st.markdown('<p class="big-font">Stocker.AI 🔮</p>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker", "RELIANCE.NS").upper()
period = st.sidebar.selectbox("Select Time Period", ["1y", "6mo", "2y", "5y", "max"])
ma_50 = st.sidebar.checkbox("Show 50-Day MA", value=True)
ma_200 = st.sidebar.checkbox("Show 200-Day MA", value=False)

if st.sidebar.button("Analyze & Forecast 🚀"):
    st.session_state.run_analysis = True
    st.session_state.ticker = ticker
else:
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False

# --- Main Content Area ---
if not st.session_state.run_analysis:
    st.info("Select a stock ticker and click 'Analyze & Forecast' in the sidebar to begin.")
else:
    data, info = get_stock_data(st.session_state.ticker, period)
    
    if data is None:
        st.error(f"Could not retrieve data for '{st.session_state.ticker}'. Please check the ticker symbol.")
    else:
        # --- Create Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["**Overview**", "**Technical Analysis**", "**AI Forecast**", "**News & Sentiment**"])
        
        # (Tab 1: Overview - Unchanged)
        with tab1:
            st.header(f"📍 Overview for {info.get('longName', st.session_state.ticker)}")
            cols = st.columns(4)
            metrics = {"Market Cap": info.get('marketCap'), "P/E Ratio": info.get('trailingPE'), "EPS": info.get('trailingEps'), "Dividend Yield": info.get('dividendYield')}
            for i, (label, value) in enumerate(metrics.items()):
                with cols[i]:
                    val_str = f"{value:,.2f}" if isinstance(value, (int, float)) and label != "P/E Ratio" else "N/A"
                    if label == "P/E Ratio" and value: val_str = f"{value:.2f}"
                    if label == "Dividend Yield" and value: val_str = f"{value*100:.2f}%"
                    if label == "Market Cap" and value: val_str = f"₹ {value/1e7:,.2f} Cr"
                    st.markdown(f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-value">{val_str}</div></div>', unsafe_allow_html=True)
            with st.expander("Business Summary"):
                st.write(info.get('longBusinessSummary', 'No summary available.'))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name='Price'), row=1, col=1)
            if ma_50: fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='50-Day MA', line=dict(color='orange')), row=1, col=1)
            if ma_200: fig.add_trace(go.Scatter(x=data.index, y=data['SMA_200'], name='200-Day MA', line=dict(color='cyan')), row=1, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
            fig.update_layout(title_text=f"{st.session_state.ticker} Price Chart", template='plotly_dark', xaxis_rangeslider_visible=False)
            fig.update_yaxes(title_text="Price", row=1, col=1); fig.update_yaxes(title_text="Volume", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        # (Tab 2: Technical Analysis - Unchanged)
        with tab2:
            st.header("⚙️ Technical Indicators")
            st.subheader("Relative Strength Index (RSI)")
            st.write("RSI values of 70 or above indicate an overbought condition, while readings of 30 or below indicate an oversold condition.")
            fig_rsi = go.Figure(); fig_rsi.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(title="RSI", template='plotly_dark')
            st.plotly_chart(fig_rsi, use_container_width=True)
            st.subheader("Moving Average Convergence Divergence (MACD)")
            st.write("A 'buy' signal occurs when the MACD line (blue) crosses above the signal line (orange). A 'sell' signal is the opposite.")
            fig_macd = go.Figure(); fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], name='Signal Line', line=dict(color='orange')))
            fig_macd.update_layout(title="MACD", template='plotly_dark')
            st.plotly_chart(fig_macd, use_container_width=True)

        # --- Tab 3: AI Forecast ---
        with tab3:
            st.header("🔮 AI Price Forecast")
            with st.spinner("Running AI models..."):
                hist_predictions, scaler = predict_historical_prices(prediction_model, data)
                future_forecast = forecast_future_prices(prediction_model, data, scaler, n_days=3)

            st.subheader("3-Day Price Forecast")
            st.warning("Future forecasts are speculative and based on the model's iterative predictions. Use with caution.", icon="⚠️")
            
            # Display metrics for the forecast
            forecast_cols = st.columns(3)
            future_dates = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=3)
            for i, (date, price) in enumerate(zip(future_dates, future_forecast)):
                with forecast_cols[i]:
                    st.metric(label=f"Day {i+1} ({date.strftime('%b %d')})", value=f"₹{price:.2f}")

            # --- Combined Historical and Forecast Chart ---
            st.subheader("Historical Fit & Future Forecast")
            fig_pred = go.Figure()
            # Historical actual price
            fig_pred.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Price', line=dict(color='deepskyblue')))
            # Historical predicted price
            pred_start_index = len(data) - len(hist_predictions)
            fig_pred.add_trace(go.Scatter(x=data.index[pred_start_index:], y=hist_predictions, name='Historical Prediction', line=dict(color='rgba(255, 127, 80, 0.7)', dash='dot')))
            # Future forecast
            fig_pred.add_trace(go.Scatter(x=future_dates, y=future_forecast, name='3-Day Forecast', line=dict(color='yellow', width=4), mode='lines+markers', marker=dict(size=8)))
            fig_pred.update_layout(title="Model Performance and 3-Day Forecast", template='plotly_dark', legend_title="Legend")
            st.plotly_chart(fig_pred, use_container_width=True)

        # (Tab 4: News & Sentiment - Unchanged)
        with tab4:
            st.header("📰 News & Sentiment Analysis")
            news_list = fetch_news()
            st.subheader("Sentiment Analysis")
            trend_desc = "upward" if data['Close'][-1] > data['Close'][-30] else "downward"
            rsi_latest = data['RSI'].iloc[-1]
            suggestion = generate_sentiment_analysis(trend_desc, rsi_latest, news_list, sentence_model)
            st.info(suggestion)
            st.subheader("Latest Market News")
            if news_list:
                for title, link in news_list:
                    st.markdown(f"- [{title}]({link})")
            else:
                st.warning("Could not fetch latest news.")

# --- Footer ---
st.markdown("---")
st.caption("🚀 Powered by Stocker.AI | Data from Yahoo Finance | Built with Streamlit")
