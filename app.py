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
import time
import json
import html

# ---------- Page Configuration ----------
st.set_page_config(page_title="Stocker.AI - Advanced Analysis", page_icon="🔮", layout="wide")

# ---------- Global Styles ----------
st.markdown("""
<style>
.big-title { font-size: 2.2rem; font-weight: 800; color: #7cc7ff; margin-bottom: 0.25rem; }
.subtle { color:#9fb3c8; font-size:0.9rem; }
.metric-box { border: 1px solid #2a7fff33; border-radius: 12px; padding: 10px; text-align: center; background: #0f1420; }
.metric-label { font-size: 0.78em; color: #c6d0e0; }
.metric-value { font-size: 1.18em; font-weight: 700; color: #e8f0ff; }
.ok { color:#22c55e; } 
.bad { color:#ef4444; } 
.warn { color:#f59e0b; }
.pill { padding: 2px 8px; border-radius: 999px; background:#1f2937; color:#cbd5e1; font-size:0.8em; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def safe_num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def inr_str(x):
    if x is None or np.isnan(x):
        return "N/A"
    return f"₹{x:,.2f}"

def pct_str(x):
    if x is None or np.isnan(x):
        return "N/A"
    return f"{x*100:.2f}%"

def normalize_nse(ticker: str) -> str:
    t = ticker.strip().upper()
    return t

# ---------- Data fetch with yfinance ----------
@st.cache_data(ttl=600, show_spinner=False)
def get_stock_data(ticker, period):
    t = yf.Ticker(ticker)
    data = t.history(period=period, auto_adjust=True)
    if data is None or data.empty:
        return None, {}

    info = {}
    try:
        fi = t.fast_info or {}
        info.update({
            "lastPrice": fi.get("last_price"),
            "currency": fi.get("currency"),
            "yearHigh": fi.get("year_high"),
            "yearLow": fi.get("year_low"),
            "marketCap": fi.get("market_cap")
        })
    except Exception:
        pass

    try:
        inf = t.info or {}
        for k in ["longName", "trailingPE", "trailingEps", "dividendYield",
                  "longBusinessSummary", "sector", "industry"]:
            if k in inf:
                info[k] = inf.get(k, None)

        if not info.get("marketCap") and "marketCap" in inf:
            info["marketCap"] = inf.get("marketCap")
    except Exception:
        pass

    return data, info

# ---------- Technical Indicators ----------
def compute_indicators(data: pd.DataFrame):
    df = data.copy()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2*df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2*df["BB_Std"]

    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum((df["High"] - df["Close"].shift()).abs(),
                   (df["Low"] - df["Close"].shift()).abs())
    )
    plus_dm = (df["High"] - df["High"].shift()).clip(lower=0)
    minus_dm = (df["Low"].shift() - df["Low"]).clip(lower=0)
    tr_n = tr.rolling(14).sum()

    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_n.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_n.replace(0, np.nan))

    dx = (abs(plus_di - minus_di) /
          (plus_di + minus_di).replace(0, np.nan)) * 100

    df["ADX"] = dx.rolling(14).mean()
    return df

def regime_description(df):
    if len(df) < 200:
        return "Insufficient data"
    trend = "Uptrend (SMA50>SMA200)" if df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1] else "Downtrend (SMA50<SMA200)"
    zone = "Trending (ADX≥25)" if df["ADX"].iloc[-1] >= 25 else "Range-bound (ADX<25)"
    return f"{trend}, {zone}"

# ---------- ML Model ----------
@st.cache_resource
def load_models():
    try:
        model = load_model("stock_prediction_model.h5")
    except Exception as e:
        st.warning(f"AI model not loaded: {e}")
        return None
    return model

def predict_historical_prices(model, data):
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(data["Close"].values.reshape(-1,1))

    win = 60
    if len(scaled) < win+1:
        return np.array([]), scaler

    X = [scaled[i-win:i,0] for i in range(win, len(scaled))]
    X = np.array(X).reshape(-1, win, 1)

    preds = model.predict(X, verbose=0)
    return scaler.inverse_transform(preds).flatten(), scaler

def forecast_future_prices(model, data, scaler, n_days):
    win = 60
    if len(data) < win+1:
        return np.array([])

    last = data["Close"].values[-win:].reshape(-1, 1)
    scaled_last = scaler.transform(last)

    future = []
    batch = scaled_last.reshape(1, win, 1)

    for _ in range(n_days):
        pred = model.predict(batch, verbose=0)[0]
        future.append(pred)
        batch = np.append(batch[:,1:,:], [[pred]], axis=1)

    return scaler.inverse_transform(np.array(future)).flatten()

# ---------- News ----------
@st.cache_data(ttl=1800)
def fetch_news():
    url = "https://www.moneycontrol.com/news/business/stocks/"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla"}, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.find_all("li", class_="clearfix", limit=10)
        return [(i.find("h2").text.strip(), i.find("a")["href"])
                for i in items if i.find("h2") and i.find("a")]
    except:
        return []

def generate_sentiment_analysis(description, rsi, news):
    pos_kw = ["growth","buy","positive","bullish","rally","profit","up"]
    neg_kw = ["decline","sell","bearish","plunge","drop","loss","down"]

    pos = sum(any(k in t.lower() for k in pos_kw) for t,_ in news)
    neg = sum(any(k in t.lower() for k in neg_kw) for t,_ in news)

    if "upward" in description:
        return "🚀 Positive trend but watch RSI." if rsi < 70 else "⚠️ Overbought zone."
    if "downward" in description:
        return "📉 Weak trend, RSI neutral." if rsi > 30 else "🔄 Oversold bounce possible."
    return "⚖ Neutral signals."

# ---------- Sidebar ----------
st.sidebar.header("⚙ Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker", "RELIANCE.NS").upper()
period = st.sidebar.selectbox("Select Time Period", ["6mo","1y","2y","5y","max"])
ma_50 = st.sidebar.checkbox("Show 50-Day MA", True)
ma_200 = st.sidebar.checkbox("Show 200-Day MA", False)
compare = st.sidebar.text_input("Compare (comma-separated tickers)", "")
forecast_days = st.sidebar.slider("Forecast Days", 1, 7, 3)
run = st.sidebar.button("Analyze & Forecast 🚀")

# ---------- Header ----------
st.markdown("<div class='big-title'>Stocker.AI 🔮</div>", unsafe_allow_html=True)
st.caption("Yahoo Finance via yfinance — Not investment advice")

# ---------- Stop early ----------
if not run:
    st.info("Select a ticker and click Analyze & Forecast")
    st.stop()

# ---------- Load Data ----------
model = load_models()
main_ticker = normalize_nse(ticker)

with st.spinner("Loading market data..."):
    data, info = get_stock_data(main_ticker, period)

if data is None:
    st.error(f"Failed to fetch data for '{main_ticker}'.")
    st.stop()

df = compute_indicators(data)


# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Technical", "AI Forecast", "Signals", "News & Compare", "LLM Chatbot"]
)

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    name = info.get('longName', main_ticker)
    st.header(f"📍 Overview for {name}")

    cols = st.columns(6)
    last_close = df["Close"].iloc[-1]
    daily_chg = df["Close"].pct_change().iloc[-1]

    with cols[0]:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">Last Close</div>'
            f'<div class="metric-value">{inr_str(last_close)}</div></div>',
            unsafe_allow_html=True
        )

    with cols[1]:
        cls = "ok" if daily_chg > 0 else "bad"
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">Daily Change</div>'
            f'<div class="metric-value {cls}">{daily_chg*100:.2f}%</div></div>',
            unsafe_allow_html=True
        )

    with cols[2]:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">RSI(14)</div>'
            f'<div class="metric-value">{safe_num(df["RSI"].iloc[-1]):.1f}</div></div>',
            unsafe_allow_html=True
        )

    with cols[3]:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">ATR(14)</div>'
            f'<div class="metric-value">{safe_num(df["ATR"].iloc[-1]):.2f}</div></div>',
            unsafe_allow_html=True
        )

    with cols[4]:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">52W High</div>'
            f'<div class="metric-value">{inr_str(safe_num(info.get("yearHigh")))}</div></div>',
            unsafe_allow_html=True
        )

    with cols[5]:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">52W Low</div>'
            f'<div class="metric-value">{inr_str(safe_num(info.get("yearLow")))}</div></div>',
            unsafe_allow_html=True
        )

    if info.get("sector") or info.get("industry"):
        st.caption(f"Sector: {info.get('sector','N/A')} • Industry: {info.get('industry','N/A')}")

    # Price chart
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7,0.3]
    )
    fig.add_trace(
        go.Candlestick(x=df.index, open=df["Open"], high=df["High"],
                       low=df["Low"], close=df["Close"], name="Price"),
        row=1, col=1
    )
    if ma_50:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50", line=dict(color="orange")), row=1, col=1)
    if ma_200:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], name="SMA 200", line=dict(color="cyan")), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)

    # Corporate Actions
    ca1, ca2 = st.columns(2)
    try:
        div = yf.Ticker(main_ticker).dividends
        if div is not None and not div.empty:
            ca1.subheader("Dividends")
            ca1.dataframe(div.tail(5).rename("Dividend"))
    except:
        pass

    try:
        sp = yf.Ticker(main_ticker).splits
        if sp is not None and not sp.empty:
            ca2.subheader("Splits")
            ca2.dataframe(sp.tail(5).rename("Split Ratio"))
    except:
        pass


# ============================================================
# TAB 2 — TECHNICAL
# ============================================================
with tab2:
    st.header("⚙ Technical Indicators")

    c1, c2 = st.columns(2)

    # RSI
    with c1:
        st.subheader("RSI (14)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    with c2:
        st.subheader("MACD")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"))
        fig_macd.update_layout(template="plotly_dark", height=280)
        st.plotly_chart(fig_macd, use_container_width=True)

    st.subheader("Volatility & Regime")
    c3, c4 = st.columns(2)

    with c3:
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(x=df.index, y=df["ATR"], name="ATR"))
        fig_atr.update_layout(template="plotly_dark", height=260)
        st.plotly_chart(fig_atr, use_container_width=True)

    with c4:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">Market Regime</div>'
            f'<div class="metric-value">{regime_description(df)}</div></div>',
            unsafe_allow_html=True
        )


# ============================================================
# TAB 3 — AI FORECAST
# ============================================================
with tab3:
    st.header("🔮 AI Price Forecast")

    if not model:
        st.warning("Model file not found.")
    else:
        with st.spinner("Running AI models..."):
            hist_preds, scaler = predict_historical_prices(model, df)
            future_preds = forecast_future_prices(model, df, scaler, forecast_days)

        st.warning("Forecasts are approximate and model-based, not financial advice.", icon="⚠")

        band = np.nan
        if len(hist_preds) > 20:
            true_vals = df["Close"].iloc[-len(hist_preds):].values
            n = min(len(true_vals), len(hist_preds))
            residuals = true_vals[-n:] - hist_preds[-n:]
            if len(residuals):
                band = float(np.nanstd(residuals))

        # Future metrics
        if future_preds.size:
            cols = st.columns(min(5, forecast_days))
            dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

            for i, (date, price) in enumerate(zip(dates, future_preds)):
                if i < len(cols):
                    with cols[i]:
                        b = f" ±{band:.2f}" if band == band else ""
                        st.metric(label=f"Day {i+1} ({date.strftime('%b %d')})", value=f"₹{price:.2f}{b}")

        # Forecast Chart
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Actual Price"))

        if len(hist_preds):
            start_idx = len(df) - len(hist_preds)
            hx = df.index[start_idx:]
            fig_f.add_trace(go.Scatter(x=hx, y=hist_preds, name="Historical Fit",
                                       line=dict(color="orange", dash="dot")))

        if future_preds.size:
            dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            fig_f.add_trace(go.Scatter(x=dates, y=future_preds,
                                       name="Forecast", line=dict(color="yellow", width=4)))

        fig_f.update_layout(template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)


# ============================================================
# TAB 4 — SIGNALS
# ============================================================
with tab4:
    st.header("📈 Signals & Simple Strategy")

    s = df.copy()
    s["Signal"] = 0
    s.loc[
        (s["MACD"] > s["MACD_Signal"]) &
        (s["MACD"].shift(1) <= s["MACD_Signal"].shift(1)),
        "Signal"
    ] = 1

    s["Position"] = s["Signal"].replace(0, np.nan).ffill().fillna(0)
    s["Return"] = s["Close"].pct_change()
    s["Strat"] = s["Position"].shift(1) * s["Return"]

    equity = (1 + s["Strat"].fillna(0)).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1)

    sharpe = (
        (np.sqrt(252) * s["Strat"].mean() / (s["Strat"].std() + 1e-9))
        if s["Strat"].std() else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    if len(s) > 252:
        cagr = equity.iloc[-1] ** (252/len(s)) - 1
    else:
        cagr = equity.iloc[-1] - 1

    c1.metric("CAGR", f"{cagr*100:.2f}%")
    c2.metric("Max Drawdown", f"{dd.min()*100:.2f}%")
    c3.metric("Sharpe-like", f"{sharpe:.2f}")
    c4.metric("Win Rate", f"{(s['Strat']>0).mean()*100:.1f}%")

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=s.index, y=equity, name="Equity", line=dict(color="#34d399")))
    fig_eq.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_eq, use_container_width=True)

    # Monthly heatmap
    m = df["Close"].resample("M").last().pct_change()
    cal = pd.DataFrame({"Year": m.index.year, "Month": m.index.month, "Return": m.values})
    pivot = cal.pivot_table(index="Year", columns="Month", values="Return", aggfunc="mean").fillna(0)

    heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.astype(str),
        colorscale="RdYlGn",
        zmin=-0.15,
        zmax=0.15
    ))
    heat.update_layout(template="plotly_dark", height=360, title="Monthly Returns Heatmap")
    st.plotly_chart(heat, use_container_width=True)


# ============================================================
# TAB 5 — NEWS & COMPARE
# ============================================================
with tab5:
    st.header("📰 News & Compare")

    news = fetch_news()

    st.subheader("Sentiment Snapshot")
    trend = "upward" if df["Close"].iloc[-1] > df["Close"].iloc[-30] else "downward"
    rsi_latest = df["RSI"].iloc[-1]

    snt = generate_sentiment_analysis(trend, rsi_latest, news)
    st.info(snt)

    st.subheader("Latest Market News")
    if news:
        for title, link in news:
            st.markdown(f"- [{title}]({link})")
    else:
        st.warning("Could not fetch latest news.")

    # Comparison
    if compare:
        tkrs = [normalize_nse(x.strip()) for x in compare.split(",") if x.strip()]
        if tkrs:
            st.subheader("Compare Close (Normalized)")
            frame = {}

            for tk in tkrs[:6]:
                try:
                    d2 = yf.Ticker(tk).history(period=period, auto_adjust=True)
                    if d2 is not None and not d2.empty:
                        ser = d2["Close"]
                        ser = ser / ser.iloc[0]
                        frame[tk] = ser
                except:
                    pass

            if frame:
                cmp_df = pd.DataFrame(frame).dropna(how="all")
                fig_cmp = go.Figure()

                for col in cmp_df.columns:
                    fig_cmp.add_trace(go.Scatter(x=cmp_df.index, y=cmp_df[col], name=col))

                fig_cmp.update_layout(template="plotly_dark", height=380)
                st.plotly_chart(fig_cmp, use_container_width=True)


# ============================================================
# TAB 6 — LLM CHATBOT
# ============================================================
with tab6:
    st.header("🤖 Stocker.AI LLM Chatbot")

    # ---------- Initialize history ----------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------- Chat Bubbles Style ----------
    st.markdown("""
    <style>
    .chat-bubble-user {
        background: linear-gradient(180deg,#0ea5b1,#0284c7);
        padding: 10px 15px;
        border-radius: 12px;
        margin: 6px 0;
        color: white;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bubble-bot {
        background: #111827;
        padding: 10px 15px;
        border-radius: 12px;
        margin: 6px 0;
        color: #e2e8f0;
        max-width: 80%;
        border: 1px solid rgba(255,255,255,0.08);
        margin-right: auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Render Chat ----------
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-bot'>{msg['content']}</div>", unsafe_allow_html=True)

    # ---------- Input Row ----------
    col_in, col_btn = st.columns([6,1])

    with col_in:
        user_msg = st.text_input(
            "Ask Stocker.AI anything about the market",
            key="chat_input_normal"
        )

    with col_btn:
        sent = st.button("Send 🚀")

    # ---------- On Send ----------
    if sent and user_msg.strip():
        user_msg = user_msg.strip()

        # Push user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_msg
        })

        # Prepare system context
        system_prompt = (
            f"You are Stocker.AI, a safe stock-market explainer model. "
            f"You DO NOT provide financial advice. You only explain stocks, indicators, volatility, AI forecasts, "
            f"and market dynamics. Stay educational. Current ticker: {main_ticker}. "
            f"Latest price: {df['Close'].iloc[-1]:.2f}. RSI: {df['RSI'].iloc[-1]:.2f}. "
            f"Market regime: {regime_description(df)}."
        )

        messages = [
            {"role": "system", "content": system_prompt},
        ] + st.session_state.chat_history

        # GROQ API CALL
        api_key = st.secrets.get("GROQ_API_KEY")
        if not api_key:
            st.error("Missing GROQ_API_KEY in Streamlit secrets.")
        else:
            import json, requests

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": messages
            }

            try:
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60
                )

                if r.status_code != 200:
                    st.error(f"Groq API Error {r.status_code}: {r.text}")
                else:
                    reply = r.json()["choices"][0]["message"]["content"]

                    # Save reply
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": reply
                    })

                    # Rerender
                    st.rerun()

            except Exception as e:
                st.error(f"API Error: {e}")

    # Clear Button
    if st.button("Clear Chat 🧹"):
        st.session_state.chat_history = []
        st.rerun()
