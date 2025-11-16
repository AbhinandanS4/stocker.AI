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
import json
import time
import html
import streamlit.components.v1 as components

# ---------------- Page config ----------------
st.set_page_config(page_title="Stocker.AI - Advanced Analysis", page_icon="🔮", layout="wide")

# ---------------- Styles ----------------
st.markdown("""
<style>
/* General */
.big-title { font-size: 2.0rem; font-weight: 800; color: #7cc7ff; margin-bottom: 0.25rem; }
.metric-box { border: 1px solid #2a7fff33; border-radius: 12px; padding: 10px; text-align: center; background: #0f1420; }
.metric-label { font-size: 0.78em; color: #c6d0e0; }
.metric-value { font-size: 1.18em; font-weight: 700; color: #e8f0ff; }

/* Tab buttons */
.tab-row { display:flex; gap:8px; margin-bottom: 12px; }
.tab-btn { background: transparent; color: #cbd5e1; padding:8px 12px; border-radius:8px; border: 1px solid transparent; cursor: pointer; }
.tab-btn-active { background: #1f2937; color: #fff; padding:8px 12px; border-radius:8px; border: 1px solid rgba(255,255,255,0.04); box-shadow: 0 1px 6px rgba(0,0,0,0.35); }

/* Chat area */
.chat-shell { display:flex; flex-direction:column; height:480px; border-radius:10px; border:1px solid rgba(255,255,255,0.03); overflow:hidden; background: #071127; padding:8px; margin-top:12px; }
.chat-history-frame { flex:1 1 auto; overflow:auto; padding:8px; }
.chat-row { display:flex; width:100%; margin:8px 0; }
.chat-bubble { padding:10px 14px; border-radius:12px; max-width:78%; line-height:1.4; font-size:0.95rem; }
.chat-user { margin-left:auto; background: linear-gradient(180deg,#0ea5b1,#0284c7); color: white; border-bottom-right-radius: 4px; }
.chat-bot { margin-right:auto; background: linear-gradient(180deg,#111827,#0f1724); color: #e6eef8; border-bottom-left-radius: 4px; border: 1px solid rgba(255,255,255,0.03); }
.chat-meta { font-size:0.78rem; color:#9fb3c8; margin-top:6px; }
.chat-input-row { display:flex; gap:8px; margin-top:8px; }
.chat-input { flex:1; }
.btn-send { padding:8px 14px; border-radius:8px; background:#2b6ef6; color:white; border:none; cursor:pointer; }

/* Minor */
.small { font-size:0.82rem; color:#9fb3c8; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ---------------- Helpers ----------------
def safe_num(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def inr_str(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"₹{x:,.2f}"

def normalize_nse(ticker: str) -> str:
    return ticker.strip().upper()

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
        for k in ["longName", "trailingPE", "trailingEps", "dividendYield", "longBusinessSummary", "sector", "industry"]:
            if k in inf:
                info[k] = inf.get(k, None)
        if not info.get("marketCap") and "marketCap" in inf:
            info["marketCap"] = inf.get("marketCap")
    except Exception:
        pass
    return data, info

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
    tr = np.maximum(df["High"]-df["Low"], np.maximum((df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()))
    plus_dm = (df["High"]-df["High"].shift()).clip(lower=0)
    minus_dm = (df["Low"].shift()-df["Low"]).clip(lower=0)
    tr_n = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr_n.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr_n.replace(0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    df["ADX"] = dx.rolling(14).mean()
    return df

def regime_description(df):
    parts = []
    if len(df) < 200:
        return "Insufficient data"
    parts.append("Uptrend (SMA50>SMA200)" if df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1] else "Downtrend (SMA50<SMA200)")
    parts.append("Trending (ADX≥25)" if df["ADX"].iloc[-1] >= 25 else "Range-bound (ADX<25)")
    return ", ".join(parts)

@st.cache_resource
def load_models():
    try:
        model = load_model("stock_prediction_model.h5")
    except Exception as e:
        st.warning(f"AI model not loaded: {e}")
        return None
    return model

def predict_historical_prices(_model, data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    window_size = 60
    if len(scaled_data) < window_size + 1:
        return np.array([]), scaler
    X_test = [scaled_data[i-window_size:i, 0] for i in range(window_size, len(scaled_data))]
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    preds = _model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds)
    return preds.flatten(), scaler

def forecast_future_prices(_model, data, scaler, n_days=3):
    window_size = 60
    if len(data) < window_size + 1:
        return np.array([])
    last_60_days = data['Close'].values[-window_size:]
    scaled_last_60_days = scaler.transform(last_60_days.reshape(-1, 1))
    future_predictions = []
    current_batch = scaled_last_60_days.reshape(1, window_size, 1)
    for _ in range(n_days):
        next_pred = _model.predict(current_batch, verbose=0)[0]
        future_predictions.append(next_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[next_pred]], axis=1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions))
    return future_predictions.flatten()

@st.cache_data(ttl=1800)
def fetch_news():
    url = "https://www.moneycontrol.com/news/business/stocks/"
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_="clearfix", limit=10)
        return [(a.find('h2').text.strip(), a.find('a')['href']) for a in articles if a.find('h2') and a.find('a')]
    except Exception:
        return []

def generate_sentiment_analysis(description, rsi_value, news_titles):
    positive_keywords = ["growth", "buy", "positive", "bullish", "rally", "profit", "up"]
    negative_keywords = ["decline", "sell", "bearish", "plunge", "drop", "loss", "down"]
    positive_score = sum(1 for title, _ in news_titles if any(word in title.lower() for word in positive_keywords))
    negative_score = sum(1 for title, _ in news_titles if any(word in title.lower() for word in negative_keywords))
    if "upward" in description:
        if rsi_value < 70:
            return f"🚀 Positive Outlook: Upward trend with RSI {rsi_value:.2f}; not overbought. News skew neutral/positive. Consider risk controls."
        else:
            return f"⚠️ Mixed Signals: Upward trend but RSI {rsi_value:.2f} near overbought. Pullback risk elevated."
    elif "downward" in description:
        if rsi_value > 30:
            return f"📉 Negative Outlook: Downward trend; RSI {rsi_value:.2f} has room to fall. Sentiment weak; caution."
        else:
            return f"⚖️ Potential Reversal: Downward trend but RSI {rsi_value:.2f} oversold—watch for bounce; high risk."
    return "⚖️ Neutral Outlook: Mixed signals. Monitor price action and volume."

# ---------------- Sidebar controls ----------------
st.sidebar.header("⚙️ Controls")
ticker = st.sidebar.text_input("Enter Stock Ticker", "RELIANCE.NS").upper()
period = st.sidebar.selectbox("Select Time Period", ["6mo", "1y", "2y", "5y", "max"])
ma_50 = st.sidebar.checkbox("Show 50-Day MA", value=True)
ma_200 = st.sidebar.checkbox("Show 200-Day MA", value=False)
compare = st.sidebar.text_input("Compare (comma-separated tickers)", "").strip()
forecast_days = st.sidebar.slider("Forecast Days", 1, 7, 3)
analyze_pressed = st.sidebar.button("Analyze & Forecast 🚀", key="analyze_btn")

# ---------------- Active tab memory ----------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 2 if analyze_pressed else 0

if analyze_pressed:
    st.session_state.active_tab = 2

# ---------------- Header + custom tab bar ----------------
st.markdown('<div class="big-title">Stocker.AI 🔮</div>', unsafe_allow_html=True)
st.caption("Yahoo Finance data via yfinance • This is not investment advice")

tab_labels = ["Overview", "Technical", "AI Forecast", "Signals", "News & Compare"]
cols = st.columns(len(tab_labels))
for i, label in enumerate(tab_labels):
    if i == st.session_state.active_tab:
        if cols[i].button(label, key=f"tab_{i}_btn"):
            st.session_state.active_tab = i
    else:
        if cols[i].button(label, key=f"tab_{i}_btn"):
            st.session_state.active_tab = i

# stop early when appropriate
if st.session_state.active_tab == 0 and not analyze_pressed:
    st.info("Select a ticker and click Analyze & Forecast to begin.")
    st.stop()

# ---------------- Main data & models ----------------
prediction_model = load_models()
main_ticker = normalize_nse(ticker)

with st.spinner("Loading market data..."):
    data, info = get_stock_data(main_ticker, period)

if data is None:
    st.error(f"Could not retrieve data for '{main_ticker}'. Please check the symbol or try another period.")
    st.stop()

df = compute_indicators(data)

# ---------------- Utility: render plots (same as before) ----------------
def show_overview():
    st.header(f"📍 Overview for {info.get('longName', main_ticker)}")
    cols = st.columns(6)
    last_close = df["Close"].iloc[-1]
    daily_chg = df["Close"].pct_change().iloc[-1]
    with cols[0]:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Last Close</div><div class="metric-value">{inr_str(last_close)}</div></div>', unsafe_allow_html=True)
    with cols[1]:
        chg_class = "ok" if daily_chg > 0 else "bad"
        st.markdown(f'<div class="metric-box"><div class="metric-label">Daily Change</div><div class="metric-value {chg_class}">{daily_chg*100:.2f}%</div></div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown(f'<div class="metric-box"><div class="metric-label">RSI(14)</div><div class="metric-value">{safe_num(df["RSI"].iloc[-1]):.1f}</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="metric-box"><div class="metric-label">ATR(14)</div><div class="metric-value">{safe_num(df["ATR"].iloc[-1]):.2f}</div></div>', unsafe_allow_html=True)
    with cols[4]:
        st.markdown(f'<div class="metric-box"><div class="metric-label">52W High</div><div class="metric-value">{inr_str(safe_num(info.get("yearHigh")))}</div></div>', unsafe_allow_html=True)
    with cols[5]:
        st.markdown(f'<div class="metric-box"><div class="metric-label">52W Low</div><div class="metric-value">{inr_str(safe_num(info.get("yearLow")))}</div></div>', unsafe_allow_html=True)

def get_chat_html():
    """
    Build chat HTML string (escaped content) and return it.
    This function returns full HTML (no outer wrapper) for injection.
    """
    parts = []
    for msg in st.session_state.chat_history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        safe = html.escape(content).replace("\n", "<br>")
        if role == "user":
            parts.append(f'''
            <div class="chat-row">
              <div class="chat-bubble chat-user">
                <div style="font-weight:600;margin-bottom:6px">You</div>
                <div>{safe}</div>
              </div>
            </div>
            ''')
        elif role == "assistant":
            parts.append(f'''
            <div class="chat-row">
              <div class="chat-bubble chat-bot">
                <div style="font-weight:600;margin-bottom:6px">Stocker.AI</div>
                <div>{safe}</div>
              </div>
            </div>
            ''')
        else:
            preview = (safe[:300] + "...") if len(safe) > 300 else safe
            parts.append(f'<div class="small">System: {preview}</div>')
    return "\n".join(parts)

def render_chat_history_component():
    """
    Inject chat HTML into an isolated component and auto-scroll.
    This avoids nested markdown wrappers and escaping.
    """
    chat_html = get_chat_html()
    # create an isolated HTML widget that contains the chat history and auto-scroll script
    injected = f"""
    <div id="st-chat-history" class="chat-history-frame" style="height:360px; overflow:auto; padding:8px;">
    {chat_html}
    </div>
    <script>
    (function() {{
        const el = document.getElementById("st-chat-history");
        if(el) {{
            el.scrollTop = el.scrollHeight;
        }}
    }})();
    </script>
    """
    # components.html renders an iframe - that's fine and prevents Streamlit from escaping markup
    components.html(injected, height=380, scrolling=True)

# ---------------- Render selected tab ----------------
if st.session_state.active_tab == 0:
    show_overview()

elif st.session_state.active_tab == 1:
    # Technical
    st.header("⚙️ Technical Indicators")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RSI")
        st.caption("70+ overbought, 30- oversold")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red"); fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(template='plotly_dark', height=280)
        st.plotly_chart(fig_rsi, use_container_width=True)
    with c2:
        st.subheader("MACD")
        st.caption("Crosses may indicate momentum shifts")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='deepskyblue')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='orange')))
        fig_macd.update_layout(template='plotly_dark', height=280)
        st.plotly_chart(fig_macd, use_container_width=True)
    st.subheader("Volatility & Regime")
    r1, r2 = st.columns(2)
    with r1:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=df.index, y=df["ATR"], name="ATR(14)", line=dict(color="#9cdcfe")))
        fig_vol.update_layout(template="plotly_dark", height=260)
        st.plotly_chart(fig_vol, use_container_width=True)
    with r2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Regime</div><div class="metric-value">{regime_description(df)}</div></div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 2:
    # ---------------- AI Forecast ----------------
    st.header("🔮 AI Price Forecast")

    if not prediction_model:
        st.warning("Model file (stock_prediction_model.h5) not found or failed to load.")
    else:
        with st.spinner("Running AI models..."):
            hist_predictions, scaler = predict_historical_prices(prediction_model, df)
            future_forecast = forecast_future_prices(prediction_model, df, scaler, n_days=forecast_days)

        st.warning("Forecasts are speculative and based on iterative predictions. Consider recent volatility and model drift.", icon="⚠️")

        band = np.nan
        if len(hist_predictions) > 20:
            true = df["Close"].iloc[-len(hist_predictions):].values
            n = min(len(true), len(hist_predictions))
            resid = true[-n:] - hist_predictions[-n:]
            if len(resid):
                band = float(np.nanstd(resid))

        if future_forecast.size:
            fc_cols = st.columns(min(5, forecast_days))
            future_dates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            for i, (date, price) in enumerate(zip(future_dates, future_forecast)):
                band_txt = f" ±{band:.2f}" if band == band else ""
                if i < len(fc_cols):
                    with fc_cols[i]:
                        st.metric(label=f"Day {i+1} ({date.strftime('%b %d')})", value=f"₹{price:.2f}{band_txt}")

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Actual Price', line=dict(color='deepskyblue')))
        if len(hist_predictions):
            pred_start_index = len(df) - len(hist_predictions)
            hx = df.index[pred_start_index:]
            fig_pred.add_trace(go.Scatter(x=hx, y=hist_predictions, name='Historical Prediction', line=dict(color='rgba(255, 127, 80, 0.8)', dash='dot')))
            if band == band:
                fig_pred.add_trace(go.Scatter(x=hx, y=hist_predictions+2*band, name='Fit +2σ', line=dict(color='rgba(255,127,80,0.3)', dash='dot')))
                fig_pred.add_trace(go.Scatter(x=hx, y=hist_predictions-2*band, name='Fit -2σ', line=dict(color='rgba(255,127,80,0.3)', dash='dot'), fill="tonexty", fillcolor="rgba(255,127,80,0.12)"))
        if future_forecast.size:
            fdates = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
            fig_pred.add_trace(go.Scatter(x=fdates, y=future_forecast, name=f'{forecast_days}-Day Forecast', line=dict(color='yellow', width=4), mode='lines+markers', marker=dict(size=8)))
            if band == band:
                fig_pred.add_trace(go.Scatter(x=fdates, y=future_forecast+2*band, name='Forecast +2σ', line=dict(color='rgba(255,255,0,0.3)', dash='dot')))
                fig_pred.add_trace(go.Scatter(x=fdates, y=future_forecast-2*band, name='Forecast -2σ', line=dict(color='rgba(255,255,0,0.3)', dash='dot'), fill="tonexty", fillcolor="rgba(255,255,0,0.12)"))
        fig_pred.update_layout(title="Model Performance and Forecast", template='plotly_dark', legend_title="Legend")
        st.plotly_chart(fig_pred, use_container_width=True)

    # ---------------- Chat UI ----------------
    st.subheader("💬 Ask Stocker.AI")

    # Chat shell (single authoritative renderer)
    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

    # Render chat history via components (avoids nested markdown problems)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    render_chat_history_component()

    # Clear chat (outside the form)
    if st.button("Clear chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.last_sent = ""
        render_chat_history_component()

    # Input form (clears on submit)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask something about the stock, forecast, or market...", key="chat_input_field")
        send = st.form_submit_button("Send")

        if send and user_input and user_input.strip():
            # stay in AI Forecast tab
            st.session_state.active_tab = 2

            # last_sent guard
            if "last_sent" not in st.session_state:
                st.session_state.last_sent = ""
            if st.session_state.last_sent == user_input.strip():
                # duplicate, skip
                render_chat_history_component()
                st.experimental_rerun()
            st.session_state.last_sent = user_input.strip()

            # system prompt (inject or refresh)
            system_prompt = (
                f"You are Stocker.AI — an assistant that explains stock market data, "
                f"technical indicators, price movements, volatility, and AI model forecasts. "
                f"You must avoid giving financial advice. "
                f"Current Ticker: {main_ticker}. "
                f"Latest Close: {df['Close'].iloc[-1]:.2f}. "
                f"RSI(14): {df['RSI'].iloc[-1]:.2f}. "
                f"Regime: {regime_description(df)}."
            )
            if not st.session_state.chat_history or st.session_state.chat_history[0].get("role") != "system":
                st.session_state.chat_history.insert(0, {"role": "system", "content": system_prompt})
            else:
                st.session_state.chat_history[0]["content"] = system_prompt

            # append user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            render_chat_history_component()

            # ---------- GROQ streaming ----------
            api_key = st.secrets.get("GROQ_API_KEY")
            if not api_key:
                st.error("Missing GROQ_API_KEY in Streamlit secrets.")
            else:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": st.session_state.chat_history,
                    "stream": True
                }

                # placeholder for typing (below the chat area)
                placeholder = st.empty()
                full_reply = ""

                try:
                    with requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, stream=True, timeout=120) as resp:
                        if resp.status_code != 200:
                            try:
                                err = resp.json()
                            except Exception:
                                err = resp.text
                            st.error(f"Groq API returned {resp.status_code}: {err}")
                        else:
                            for raw in resp.iter_lines(decode_unicode=True):
                                if not raw or not raw.startswith("data:"):
                                    continue
                                data = raw[5:].strip()
                                if data == "[DONE]":
                                    break
                                try:
                                    j = json.loads(data)
                                    choices = j.get("choices") or []
                                    token = ""
                                    if choices:
                                        delta = choices[0].get("delta", {}) or {}
                                        token = delta.get("content", "") or ""
                                        if token == "":
                                            # sometimes the message block is sent
                                            token = choices[0].get("message", {}).get("content", "") or ""
                                except Exception:
                                    token = data

                                if token:
                                    full_reply += token
                                    # show typing in placeholder
                                    safe_partial = html.escape(full_reply).replace("\n", "<br>")
                                    placeholder.markdown(f'''
                                    <div class="chat-row">
                                      <div class="chat-bubble chat-bot typing">
                                        <div style="font-weight:600;margin-bottom:6px">Stocker.AI</div>
                                        <div>{safe_partial}▌</div>
                                      </div>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                    # keep chat history rendered and scroll
                                    render_chat_history_component()
                                    time.sleep(0.02)

                            # finalize
                            placeholder.empty()
                except requests.exceptions.RequestException as e:
                    st.error(f"Groq API request failed: {e}")

                # Append final assistant reply (if not duplicate)
                last_assistant = None
                for m in reversed(st.session_state.chat_history):
                    if m.get("role") == "assistant":
                        last_assistant = m.get("content")
                        break
                if last_assistant != full_reply:
                    st.session_state.chat_history.append({"role": "assistant", "content": full_reply})

                # final re-render
                render_chat_history_component()

    # close chat shell
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.active_tab == 3:
    # Signals
    st.header("📈 Simple Signals & Stats")
    s = df.copy()
    s["Signal"] = 0
    s.loc[(s["MACD"] > s["MACD_Signal"]) & (s["MACD"].shift(1) <= s["MACD_Signal"].shift(1)), "Signal"] = 1
    s["Position"] = s["Signal"].replace(0, np.nan).ffill().fillna(0)
    s["Return"] = s["Close"].pct_change()
    s["Strat"] = s["Position"].shift(1) * s["Return"]
    equity = (1 + s["Strat"].fillna(0)).cumprod()
    peak = equity.cummax()
    dd = (equity/peak - 1)
    sharpe_like = (np.sqrt(252) * s["Strat"].mean() / (s["Strat"].std() + 1e-9)) if s["Strat"].std() else 0.0
    c1, c2, c3, c4 = st.columns(4)
    if len(s) > 252:
        cagr = (equity.iloc[-1])**(252/len(s)) - 1
    else:
        cagr = equity.iloc[-1] - 1
    c1.metric("CAGR", f"{cagr*100:,.2f}%")
    c2.metric("Max Drawdown", f"{dd.min()*100:,.2f}%")
    c3.metric("Sharpe-like", f"{sharpe_like:.2f}")
    c4.metric("Win Rate", f"{(s['Strat']>0).mean()*100:,.1f}%")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(x=s.index, y=equity, name="Equity", line=dict(color="#34d399")))
    fig_eq.update_layout(template="plotly_dark", height=300, title="Strategy Equity (MACD cross long-only)")
    st.plotly_chart(fig_eq, use_container_width=True)

elif st.session_state.active_tab == 4:
    # News & Compare
    st.header("📰 News & Compare")
    news_list = fetch_news()
    st.subheader("Sentiment Snapshot")
    trend_desc = "upward" if df['Close'][-1] > df['Close'][-30] else "downward" if len(df) > 30 else "neutral"
    rsi_latest = df['RSI'].iloc[-1]
    suggestion = generate_sentiment_analysis(trend_desc, rsi_latest, news_list)
    st.info(suggestion)
    st.subheader("Latest Market News")
    if news_list:
        for title, link in news_list:
            st.markdown(f"- [{title}]({link})")
    else:
        st.warning("Could not fetch latest news.")
    if compare:
        comp_list = [normalize_nse(x.strip()) for x in compare.split(",") if x.strip()]
        if comp_list:
            st.subheader("Compare Close (Normalized)")
            frame = {}
            for tkr in comp_list[:6]:
                try:
                    d2 = yf.Ticker(tkr).history(period=period, auto_adjust=True)
                    if d2 is not None and not d2.empty:
                        ser = d2["Close"]
                        ser = ser / ser.iloc[0]
                        frame[tkr] = ser
                except Exception:
                    pass
            if frame:
                cmp_df = pd.DataFrame(frame).dropna(how="all")
                fig_cmp = go.Figure()
                for col in cmp_df.columns:
                    fig_cmp.add_trace(go.Scatter(x=cmp_df.index, y=cmp_df[col], name=col))
                fig_cmp.update_layout(template='plotly_dark', height=380, title="Normalized Performance")
                st.plotly_chart(fig_cmp, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.caption("🚀 Powered by Stocker.AI | Data from Yahoo Finance via yfinance | Built with Streamlit")
