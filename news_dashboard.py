import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from textblob import TextBlob
import time

# --------------------------- SETTINGS ---------------------------
st.set_page_config(
    page_title="Finance News & Insights",
    layout="wide",
    page_icon="📰",
)

API_KEY = st.secrets.get("NEWS_API_KEY", "")  # Store securely in Streamlit Secrets

# --------------------------- FUNCTIONS ---------------------------

def fetch_news(query="India finance", page_size=10):
    """
    Fetch news using NewsAPI or fallback to Google RSS
    """
    articles = []
    try:
        if API_KEY:
            url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize={page_size}&apiKey={API_KEY}"
            res = requests.get(url)
            data = res.json()
            if data.get("articles"):
                articles = data["articles"]
    except Exception as e:
        st.warning(f"⚠️ NewsAPI error: {e}")
    return articles


def fetch_market_indices():
    """
    Fetch real-time market indices (India + global)
    """
    indices = {
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "NASDAQ": "^IXIC",
        "DOW JONES": "^DJI",
        "S&P 500": "^GSPC",
    }
    data = []
    try:
        import yfinance as yf
        for name, symbol in indices.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                latest, prev = hist["Close"].iloc[-1], hist["Close"].iloc[-2]
                change = ((latest - prev) / prev) * 100
                data.append((name, latest, change))
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
    return data


def sentiment_score(text):
    """
    Basic sentiment analysis
    """
    return TextBlob(text).sentiment.polarity


def personalized_recommendation(news_list):
    """
    Learn what user reads — simulate recommendation
    """
    keywords = []
    for n in news_list:
        if "title" in n and any(w in n["title"].lower() for w in ["rbi", "bank", "inflation", "startup", "budget", "policy"]):
            keywords.append("Finance")
        elif "stock" in n["title"].lower():
            keywords.append("Stock Market")
        elif "gov" in n["title"].lower() or "modi" in n["title"].lower():
            keywords.append("Government Updates")
    return pd.Series(keywords).value_counts().head(3).index.tolist()


# --------------------------- SIDEBAR ---------------------------

st.sidebar.header("⚙️ Controls")
query = st.sidebar.text_input("Search topic", "India finance")
refresh_interval = st.sidebar.selectbox("Auto-refresh interval", ["2 min", "5 min", "10 min"])
page_size = st.sidebar.slider("Number of headlines", 5, 30, 10)
manual = st.sidebar.button("🔄 Refresh Now")

# --------------------------- MAIN CONTENT ---------------------------

st.title("📰 Finance News & Insights")
st.markdown("### Live updates from Indian business, markets, and policy.")

col1, col2 = st.columns([2, 1])

# NEWS SECTION
with col1:
    st.subheader("📢 Top News")
    news_data = fetch_news(query, page_size)

    if not news_data:
        st.warning("No articles found at the moment. Try again later.")
    else:
        for n in news_data:
            title = n.get("title", "")
            url = n.get("url", "#")
            desc = n.get("description", "")
            sent = sentiment_score(title)
            sentiment_text = (
                "🟢 Positive" if sent > 0.2 else "🔴 Negative" if sent < -0.2 else "🟡 Neutral"
            )

            st.markdown(
                f"""
                <div style='background-color:#F4F6F8; padding:10px; border-radius:10px; margin-bottom:8px'>
                    <b><a href='{url}' target='_blank' style='text-decoration:none; color:#1A5276'>{title}</a></b><br>
                    <small>{desc}</small><br>
                    <i>{sentiment_text}</i>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Personalized Recommendations
        st.markdown("---")
        recs = personalized_recommendation(news_data)
        if recs:
            st.markdown(f"**🧠 Suggested Topics You Might Like:** {' | '.join(recs)}")


# MARKET INDICES
with col2:
    st.subheader("📊 Market Snapshot")
    market_data = fetch_market_indices()

    for name, latest, change in market_data:
        color = "green" if change > 0 else "red"
        st.markdown(
            f"<div style='background-color:#EBF5FB; border-left:5px solid {color}; padding:8px; margin-bottom:6px'>"
            f"<b>{name}</b><br>{latest:,.2f} <span style='color:{color}'>({change:+.2f}%)</span>"
            "</div>",
            unsafe_allow_html=True
        )

# --------------------------- REFRESH ---------------------------
if manual:
    st.experimental_rerun()

interval_sec = int(refresh_interval.split()[0]) * 60
st_autorefresh = st.experimental_data_editor if interval_sec else None

# --------------------------- FOOTER ---------------------------
st.markdown("<br><hr><center>🧭 Built for real-time market insight & trend discovery.</center>", unsafe_allow_html=True)
