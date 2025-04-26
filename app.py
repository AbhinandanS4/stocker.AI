import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import requests
from bs4 import BeautifulSoup
import spacy
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import subprocess
import time

# Custom CSS for animations and styling
st.markdown("""
    <style>
    .big-font {
        font-size:36px !important;
        font-weight:bold;
        color:#1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .subtitle-font {
        font-size:22px !important;
        font-weight:600;
        color:#333333;
        margin-top:10px;
    }
    body {
        background: linear-gradient(270deg, #1a2a6c, #b21f1f, #ff6a00);
        background-size: 400% 400%;
        animation: gradient 20s ease infinite;
        color: white;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Stocker.AI Title
st.markdown('<p class="big-font">📈 Welcome to Stocker.AI 🚀</p>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    model = load_model("your_model.h5")  # Adjust path if needed
    return model, nlp

model, nlp = load_models()

# Prediction function
def predict_stock(ticker, model, period):
    stock_data = yf.download(ticker, period=period, auto_adjust=True)
    stock_close = stock_data[['Close']].values
    scaler = MinMaxScaler()
    stock_close_scaled = scaler.fit_transform(stock_close)
    window_size = 60

    X = []
    for i in range(window_size, len(stock_close_scaled)):
        X.append(stock_close_scaled[i - window_size:i, 0])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    return predictions, stock_data, scaler

# Plotting function
def plot_predictions(predicted, actual, ticker):
    df = pd.DataFrame({
        'Actual': actual.flatten(),
        'Predicted': predicted.flatten()
    })

    fig = make_subplots(specs=[[{'secondary_y': False}]])
    fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Predicted'], name='Predicted', line=dict(color='red')))
    fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Time', yaxis_title='Price')
    return fig

# Fetch news function
def fetch_news():
    url = "https://www.moneycontrol.com/news/business/stocks/"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    articles = soup.find_all('li', class_="clearfix")
    news_list = []
    for article in articles:
        title_tag = article.find('h2')
        link_tag = article.find('a', href=True)
        if title_tag and link_tag:
            news_list.append((title_tag.text.strip(), link_tag['href']))
    return news_list

# Investment suggestion function (simple)
def suggest_investment(description, news_titles):
    positive_words = ["growth", "buy", "positive", "bullish", "rally"]
    negative_words = ["decline", "sell", "bearish", "plunge", "drop"]

    positive_score = sum(word in str(news_titles).lower() for word in positive_words)
    negative_score = sum(word in str(news_titles).lower() for word in negative_words)

    if "upward" in description and positive_score >= negative_score:
        return "🚀 Positive trend ahead! A good opportunity to invest (but still DYOR)."
    elif "downward" in description or negative_score > positive_score:
        return "⚠️ The Model showed a Negative trend; Caution advised. Market sentiment seems weak."
    else:
        return "⚖️ Neutral outlook. Monitor news and indicators closely."

# --- Main App ---

# Dropdown for selecting the time period
period = st.selectbox("Select Time Period for Data:", ["6mo", "1y"])

ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS, INFY.NS, RELIANCE.NS)", "RELIANCE.NS")

if st.button("Predict & Analyze 🚀"):
    with st.spinner('Fetching stock data and news...please wait'):
        time.sleep(1)

        predictions, stock_data, scaler = predict_stock(ticker, model, period)  # Use selected period here
        actual_prices = stock_data['Close'].values[-len(predictions):]

        news = fetch_news()

    # Expandable prediction section
    with st.expander("📈 Stock Price Prediction Details"):
        st.plotly_chart(plot_predictions(predictions, actual_prices, ticker), use_container_width=True)
        st.success(f"Prediction for {ticker} ready!")

    # Expandable news section
    with st.expander("📰 Latest Stock News"):
        if news:
            for title, link in news[:5]:  # Show top 5 news
                st.markdown(f"- [{title}]({link})")
        else:
            st.warning("No recent news found.")

    # Investment Suggestion
    with st.expander("🧠 Investment Analysis"):
        trend_desc = "upward" if predictions[-1] > predictions[0] else "downward"
        suggestion = suggest_investment(trend_desc, news)
        st.info(suggestion)

# Footer
st.markdown("---")
st.caption("🚀 Powered by stocker.AI | Built with ❤️ using Streamlit, yfinance, and TensorFlow")
