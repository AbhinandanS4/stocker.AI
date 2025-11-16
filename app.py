import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import requests
import json
import html
import time
from bs4 import BeautifulSoup
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="Stocker.AI - Advanced Analysis",
    page_icon="🔮",
    layout="wide"
)

# ------------------------------------------------------------
# Global CSS Styles (Discord-style chat, modern input bar)
# ------------------------------------------------------------
st.markdown("""
<style>

body {
    font-family: 'Inter', sans-serif;
}

/* Title */
.big-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #7cc7ff;
    margin-bottom: 0.3rem;
}

/* Metric Boxes */
.metric-box {
    border: 1px solid rgba(122,162,247,0.22);
    background: var(--background-color, #0f1420);
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}
.metric-label {
    font-size: 0.78em;
    color: #c6d0e0;
}
.metric-value {
    font-size: 1.18em;
    font-weight: 700;
    color: #e8f0ff;
}

/* Tab buttons */
.tab-row { display:flex; gap:8px; margin-bottom: 15px; }
.tab-btn {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.04);
    padding: 8px 14px;
    border-radius: 8px;
    cursor: pointer;
    color: #cbd5e1;
}
.tab-btn-active {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    box-shadow: 0 1px 6px rgba(0,0,0,0.35);
}

/* Chat container (collapsible inside forecast tab) */
.chat-panel {
    width: 100%;
    background: rgba(12,17,28,0.65);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px;
    margin-top: 14px;
}

.chat-history {
    height: 420px;
    overflow-y: auto;
    padding: 8px;
    scroll-behavior: smooth;
}

/* Discord-style bubbles */
.message-block {
    width: 100%;
    margin: 10px 0;
    padding: 10px 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    color: #e8ecf2;
}
.message-username {
    font-weight: 700;
    margin-bottom: 4px;
    color: #7cc7ff;
}

/* Input bar */
.input-bar-wrapper {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
}

.input-bar {
    flex: 1;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 10px 14px;
    border-radius: 10px;
    color: white;
    outline: none;
}

.send-btn {
    background: #2b6ef6;
    border: none;
    padding: 10px 16px;
    border-radius: 10px;
    cursor: pointer;
    font-size: 18px;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def safe_num(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

def inr_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"₹{x:,.2f}"

def normalize_ticker(t):
    return t.strip().upper()

# ------------------------------------------------------------
# Cached Yahoo Finance Data
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def get_stock_data(ticker, period):
    t = yf.Ticker(ticker)
    data = t.history(period=period, auto_adjust=True)
    if data.empty:
        return None, {}

    info = {}

    # fast_info
    try:
        fi = t.fast_info or {}
        info.update({
            "lastPrice": fi.get("last_price"),
            "currency": fi.get("currency"),
            "yearHigh": fi.get("year_high"),
            "yearLow": fi.get("year_low"),
            "marketCap": fi.get("market_cap")
        })
    except:
        pass

    # info
    try:
        inf = t.info or {}
        for k in ["longName", "trailingPE", "trailingEps", "dividendYield",
                  "longBusinessSummary", "sector", "industry"]:
            if k in inf:
                info[k] = inf.get(k)
    except:
        pass

    return data, info

# ------------------------------------------------------------
# Indicators
# ------------------------------------------------------------
def compute_indicators(df):
    df = df.copy()

    df["SMA_50"]  = df["Close"].rolling(50).mean()
    df["SMA_200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100/(1+rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

    df["BB_Mid"] = df["Close"].rolling(20).mean()
    df["BB_Std"] = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2*df["BB_Std"]
    df["BB_Lower"] = df["BB_Mid"] - 2*df["BB_Std"]

    return df

def regime_description(df):
    if len(df) < 200:
        return "Insufficient data"
    trend = "Uptrend" if df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1] else "Downtrend"
    vol   = "Trending" if df["ATR"].iloc[-1] > df["ATR"].mean() else "Range-bound"
    return f"{trend}, {vol}"

# ------------------------------------------------------------
# Load AI Model for Forecast
# ------------------------------------------------------------
@st.cache_resource
def load_forecast_model():
    try:
        return load_model("stock_prediction_model.h5")
    except:
        return None

# ------------------------------------------------------------
# Forecast functions
# ------------------------------------------------------------
def predict_history(model, df):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(df["Close"].values.reshape(-1,1))
    ws = 60
    if len(scaled) < ws:
        return np.array([]), scaler

    X = []
    for i in range(ws, len(scaled)):
        X.append(scaled[i-ws:i,0])
    X = np.array(X).reshape(-1, ws, 1)

    preds = model.predict(X, verbose=0)
    preds = scaler.inverse_transform(preds)
    return preds.flatten(), scaler

def forecast_future(model, df, scaler, days):
    ws = 60
    if len(df) < ws:
        return np.array([])

    last = df["Close"].values[-ws:]
    scaled = scaler.transform(last.reshape(-1,1))
    batch = scaled.reshape(1, ws, 1)

    results = []
    for _ in range(days):
        nxt = model.predict(batch, verbose=0)[0]
        results.append(nxt)
        batch = np.append(batch[:,1:,:], [[nxt]], axis=1)

    return scaler.inverse_transform(np.array(results)).flatten()

# ------------------------------------------------------------
# Fetch News
# ------------------------------------------------------------
@st.cache_data(ttl=1800)
def fetch_news():
    url = "https://www.moneycontrol.com/news/business/stocks/"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        items = soup.find_all("li", class_="clearfix", limit=10)
        return [(i.find("h2").text.strip(), i.find("a")["href"])
                for i in items if i.find("h2") and i.find("a")]
    except:
        return []

# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

ticker_input = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")
period = st.sidebar.selectbox("Period", ["6mo","1y","2y","5y","max"])
ma50 = st.sidebar.checkbox("50-day MA", True)
ma200 = st.sidebar.checkbox("200-day MA", False)
compare = st.sidebar.text_input("Compare With", "")
forecast_days = st.sidebar.slider("Forecast Days", 1, 7, 3)
run_forecast = st.sidebar.button("Analyze & Forecast 🚀")

# Track active tab state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 2 if run_forecast else 0
if run_forecast:
    st.session_state.active_tab = 2

# Title
st.markdown('<div class="big-title">Stocker.AI 🔮</div>', unsafe_allow_html=True)
st.caption("Enhanced forecasting • Not financial advice")


# ------------------------------------------------------------
# Fetch stock data
# ------------------------------------------------------------
main_ticker = normalize_ticker(ticker_input)
with st.spinner("Fetching market data..."):
    df_raw, info = get_stock_data(main_ticker, period)

if df_raw is None:
    st.error("Invalid ticker or no data available.")
    st.stop()

df = compute_indicators(df_raw)
model = load_forecast_model()

# ------------------------------------------------------------
# Tab Bar (manual controls)
# ------------------------------------------------------------
tab_names = ["Overview", "Technical", "AI Forecast", "Signals", "News & Compare"]
cols = st.columns(len(tab_names))

for i, label in enumerate(tab_names):
    if i == st.session_state.active_tab:
        if cols[i].button(label, key=f"tab_{i}_active"):
            st.session_state.active_tab = i
    else:
        if cols[i].button(label, key=f"tab_{i}"):
            st.session_state.active_tab = i

# ------------------------------------------------------------
# Helper render functions for Overview + Technical
# ------------------------------------------------------------
def show_overview():
    st.header(f"📍 Overview: {info.get('longName', main_ticker)}")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    last = df["Close"].iloc[-1]
    chg = df["Close"].pct_change().iloc[-1]

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Last Close</div>
            <div class="metric-value">{inr_str(last)}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        color = "ok" if chg>0 else "bad"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Daily Change</div>
            <div class="metric-value" style="color:{'#22c55e' if chg>0 else '#ef4444'}">
                {chg*100:.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">RSI(14)</div>
            <div class="metric-value">{df['RSI'].iloc[-1]:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">ATR(14)</div>
            <div class="metric-value">{df['ATR'].iloc[-1]:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">52W High</div>
            <div class="metric-value">{inr_str(info.get("yearHigh"))}</div>
        </div>
        """, unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">52W Low</div>
            <div class="metric-value">{inr_str(info.get("yearLow"))}</div>
        </div>
        """, unsafe_allow_html=True)

    st.caption(f"Sector: {info.get('sector','N/A')} • Industry: {info.get('industry','N/A')}")

    # Price chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.7, 0.3])

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price"
        ), row=1, col=1
    )
    if ma50:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], name="SMA 50",
                                 line=dict(color="orange")), row=1, col=1)
    if ma200:
        fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], name="SMA 200",
                                 line=dict(color="cyan")), row=1, col=1)

    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color="lightskyblue"), row=2, col=1)

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False,
                      height=550, title=f"{main_ticker} – Price Chart")
    st.plotly_chart(fig, use_container_width=True)


def show_technical():
    st.header("⚙️ Technical Indicators")

    left, right = st.columns(2)

    with left:
        st.subheader("RSI")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        fig.add_hline(y=70, line_dash="dot", line_color="red")
        fig.add_hline(y=30, line_dash="dot", line_color="green")
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("MACD")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Regime
    st.subheader("Regime")
    st.info(regime_description(df))


# ------------------------------------------------------------
# AI Forecast Tab (Graphs + Collapsible Chat Panel)
# ------------------------------------------------------------
if st.session_state.active_tab == 2:

    st.header("🔮 AI Forecast")

    if not model:
        st.error("AI model file missing: stock_prediction_model.h5")
    else:
        with st.spinner("Generating forecast..."):
            hist_pred, scaler = predict_history(model, df)
            future = forecast_future(model, df, scaler, forecast_days)

        st.caption("⚠ Forecasts are approximate and for educational use only.")

        # Forecast metrics
        dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1),
                               periods=forecast_days)
        cols = st.columns(forecast_days)

        for i, (d, p) in enumerate(zip(dates, future)):
            with cols[i]:
                st.metric(f"Day {i+1} ({d.strftime('%b %d')})", f"₹{p:.2f}")

        # Forecast plot
        figf = go.Figure()
        figf.add_trace(
            go.Scatter(x=df.index, y=df["Close"], name="Actual",
                       line=dict(color="deepskyblue"))
        )

        if len(hist_pred):
            idx = df.index[-len(hist_pred):]
            figf.add_trace(
                go.Scatter(x=idx, y=hist_pred, name="Model Fit",
                           line=dict(color="orange", dash="dot"))
            )

        figf.add_trace(
            go.Scatter(x=dates, y=future, name="Forecast",
                       line=dict(color="yellow"), mode="lines+markers")
        )

        figf.update_layout(template="plotly_dark", height=520,
                           title="Model Forecast")
        st.plotly_chart(figf, use_container_width=True)

    # --------------------------------------------------------
    # COLLAPSIBLE CHATBOT PANEL (F3)
    # --------------------------------------------------------
    with st.expander("💬 Ask Stocker.AI", expanded=False):

        # Chat wrapper
        st.markdown('<div class="chat-panel">', unsafe_allow_html=True)

        # Chat history container (empty; we fill via JS)
        st.markdown("""
            <div id="chat-history" class="chat-history"></div>
        """, unsafe_allow_html=True)

        # Input bar (no form)
        st.markdown("""
            <div class="input-bar-wrapper">
                <input id="chat-input" class="input-bar" placeholder="Ask something about the stock...">
                <button id="send-btn" class="send-btn">➤</button>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------
# Signals Tab
# ------------------------------------------------------------
if st.session_state.active_tab == 3:
    st.header("📈 Simple Signals & Strategy")

    s = df.copy()
    s["Signal"] = (s["MACD"] > s["MACD_Signal"]).astype(int)
    s["Position"] = s["Signal"].replace(0, np.nan).ffill().fillna(0)
    s["Return"] = s["Close"].pct_change()
    s["Strat"] = s["Position"].shift(1) * s["Return"]
    equity = (1 + s["Strat"].fillna(0)).cumprod()

    st.metric("Sharpe-like", (np.sqrt(252)*s["Strat"].mean()/(s["Strat"].std()+1e-9)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=equity,
                             line=dict(color="#34d399"), name="Equity"))
    fig.update_layout(template="plotly_dark", height=400,
                      title="MACD Strategy Equity")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# News & Compare Tab
# ------------------------------------------------------------
if st.session_state.active_tab == 4:
    st.header("📰 News & Comparison")

    news = fetch_news()
    if news:
        for title, link in news:
            st.markdown(f"- [{title}]({link})")
    else:
        st.warning("No news available.")

    if compare:
        tickers = [normalize_ticker(x) for x in compare.split(",")]
        frame = {}
        for t in tickers:
            try:
                d2 = yf.Ticker(t).history(period=period, auto_adjust=True)
                if not d2.empty:
                    frame[t] = d2["Close"] / d2["Close"].iloc[0]
            except:
                pass

        if frame:
            cmp_df = pd.DataFrame(frame)
            fig = go.Figure()
            for col in cmp_df.columns:
                fig.add_trace(go.Scatter(x=cmp_df.index, y=cmp_df[col], name=col))
            fig.update_layout(template="plotly_dark", height=420,
                              title="Normalized Comparison")
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------
# Chat State
# ------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_user_msg" not in st.session_state:
    st.session_state.last_user_msg = None

# ------------------------------------------------------------
# Render chat history as HTML (returned, not printed)
# ------------------------------------------------------------
def build_chat_html():
    html_blocks = []

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = html.escape(msg["content"]).replace("\n", "<br>")

        username = "Stocker.AI" if role == "assistant" else "You"
        html_blocks.append(f"""
        <div class="message-block">
            <div class="message-username">{username}</div>
            <div class="message-content">{content}</div>
        </div>
        """)

    return "".join(html_blocks)


# ------------------------------------------------------------
# Inject chat history into browser DOM
# ------------------------------------------------------------
def push_chat_to_browser():
    chat_html = build_chat_html().replace("`", "\\`")

    js = f"""
    <script>
        const box = window.parent.document.getElementById("chat-history");
        if (box) {{
            box.innerHTML = `{chat_html}`;
            box.scrollTop = box.scrollHeight;
        }}
    </script>
    """
    st.components.v1.html(js, height=0)


# Push history on page load (if forecast tab active)
if st.session_state.active_tab == 2:
    push_chat_to_browser()


# ------------------------------------------------------------
# Python → Groq streaming function
# ------------------------------------------------------------
def stream_from_groq():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return "Groq API key missing."

    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": st.session_state.chat_history,
        "stream": True
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    reply = ""
    try:
        with requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            json=payload, headers=headers, stream=True, timeout=60
        ) as resp:

            if resp.status_code != 200:
                try:
                    return f"Groq error: {resp.json()}"
                except:
                    return f"Groq error: {resp.text}"

            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    j = json.loads(data)
                    token = j["choices"][0].get("delta", {}).get("content", "")
                except:
                    token = data

                if token:
                    reply += token

        return reply

    except Exception as e:
        return f"Groq request failed: {e}"


# ------------------------------------------------------------
# System Prompt (always at index 0)
# ------------------------------------------------------------
def ensure_system_prompt():
    prompt = (
        f"You are Stocker.AI, a financial analysis assistant. "
        f"You explain indicators, price action, volatility, and forecasts. "
        f"You DO NOT give trading advice. "
        f"Ticker: {main_ticker}. Latest Close: {df['Close'].iloc[-1]:.2f}. "
        f"RSI: {df['RSI'].iloc[-1]:.2f}. Regime: {regime_description(df)}."
    )

    if not st.session_state.chat_history:
        st.session_state.chat_history.append({"role": "system", "content": prompt})
    else:
        st.session_state.chat_history[0] = {"role": "system", "content": prompt}


# ------------------------------------------------------------
# Handle incoming JS → Python message
# ------------------------------------------------------------
message = st.experimental_get_query_params().get("new_msg", None)
if message:
    message = message[0]

    # Prevent duplicates on rerun
    if message != st.session_state.last_user_msg:

        ensure_system_prompt()

        st.session_state.chat_history.append({"role": "user", "content": message})
        st.session_state.last_user_msg = message

        push_chat_to_browser()

        # Call Groq
        assistant_reply = stream_from_groq()

        st.session_state.chat_history.append(
            {"role": "assistant", "content": assistant_reply}
        )

        push_chat_to_browser()

    # Remove URL param to avoid repeat event
    st.experimental_set_query_params()


# ------------------------------------------------------------
# JavaScript send-button handler
# ------------------------------------------------------------
if st.session_state.active_tab == 2:
    js_logic = """
    <script>
        const btn = window.parent.document.getElementById("send-btn");
        const input = window.parent.document.getElementById("chat-input");

        function sendMsg() {
            const msg = input.value.trim();
            if (!msg) return;

            const base = window.location.href.split("?")[0];
            window.location.href = base + "?new_msg=" + encodeURIComponent(msg);
        }

        if (btn) btn.onclick = sendMsg;

        if (input) {
            input.addEventListener("keydown", function(e){
                if (e.key === "Enter") sendMsg();
            });
        }
    </script>
    """
    st.components.v1.html(js_logic, height=0)


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.caption("🚀 Powered by Stocker.AI • Built with Streamlit & Groq • Yahoo Finance data")
