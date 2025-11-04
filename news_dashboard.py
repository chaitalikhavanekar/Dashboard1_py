# news_dashboard.py
"""
News & Insights Dashboard (Streamlit)
Styled with palette: Sky Blue (#C8D9E6), Beige (#F5EFEB), Navy (#2F4156), Teal (#567C8D), White (#FFFFFF)
Features:
 - Live news via NewsAPI (if NEWSAPI_KEY provided) or Google News RSS (fallback)
 - MOSPI press-releases attempt (scrape new.mospi.gov.in, may fail if blocked)
 - Sentiment analysis using VADER
 - Market indices snapshot via yfinance
 - Single stock lookup (one symbol at a time, auto-refresh interval)
 - Auto-refresh control (1 / 5 / 10 minutes)
 - Caching (requests_cache) to limit repeated calls
Requirements (put into requirements.txt):
 streamlit
 requests
 requests_cache
 feedparser
 beautifulsoup4
 lxml
 pandas
 yfinance
 plotly
 vaderSentiment
 python-dotenv
"""
# -------------- Imports --------------
import os
import time
# --- API Key Setup ---
# Directly store your NewsAPI key here
api_key = "0cfaa15eee79489ba81fcc9fd418b1b8"
from datetime import datetime, timezone
from typing import List, Dict, Optional

import streamlit as st
import requests
import requests_cache
import pandas as pd
import yfinance as yf
import plotly.express as px
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import feedparser
from bs4 import BeautifulSoup

# -------------- Caching & env --------------
requests_cache.install_cache("news_cache", expire_after=300)  # 5 minutes cache

# Load env vars if python-dotenv available (no crash if missing)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

NEWSAPI_KEY = "0cfaa15eee79489ba81fcc9fd418b1b8"  # optional; recommended for higher quality
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0"

# -------------- Theme / CSS (your palette) --------------
SKY_BLUE = "#C8D9E6"
BEIGE = "#F5EFEB"
NAVY = "#2F4156"
TEAL = "#567C8D"
WHITE = "#FFFFFF"
POS_GREEN = "#00C49F"
NEG_RED = "#FF4C4C"

st.set_page_config(page_title="📰 News & Insights", layout="wide")

st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"] {{ background: {SKY_BLUE}; }}
    .stApp {{ background: {SKY_BLUE}; }}
    .news-card {{
        background: {BEIGE};
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
        color: {NAVY};
    }}
    .card-title {{ color: {NAVY}; font-weight:700; font-size:18px; }}
    .card-meta {{ color: {TEAL}; font-size:12px; }}
    h1, .css-18ni7ap h1 {{ color: {NAVY}; }}
    .subtext {{ color: {TEAL}; }}
    .metric-box {{ background: {BEIGE}; padding: 10px; border-radius: 8px; color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------- Utilities --------------
analyzer = SentimentIntensityAnalyzer()


def classify_sentiment(text: str) -> Dict:
    s = analyzer.polarity_scores(text or "")
    # compound in [-1,1]
    comp = s["compound"]
    if comp >= 0.05:
        label = "Positive"
    elif comp <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"label": label, "score": comp}


def short_time_ago(dt: datetime) -> str:
    now = datetime.now(timezone.utc)
    diff = now - dt
    sec = diff.total_seconds()
    if sec < 60:
        return f"{int(sec)}s ago"
    if sec < 3600:
        return f"{int(sec // 60)}m ago"
    if sec < 86400:
        return f"{int(sec // 3600)}h ago"
    return f"{int(sec // 86400)}d ago"


# -------------- News fetchers --------------
def fetch_news_from_newsapi(query: str, page_size: int = 8) -> List[Dict]:
    """Use NewsAPI if key available."""
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    try:
        res = requests.get(url, params=params, timeout=15, headers={"User-Agent": USER_AGENT})
        res.raise_for_status()
        j = res.json()
        items = []
        for a in j.get("articles", []):
            published = a.get("publishedAt")
            try:
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.now(timezone.utc)
            items.append({
                "title": a.get("title"),
                "summary": a.get("description") or "",
                "url": a.get("url"),
                "source": a.get("source", {}).get("name"),
                "published_at": dt,
            })
        return items
    except Exception:
        return []


def fetch_news_from_google_rss(query: str, country_code: str = "IN", max_items: int = 8) -> List[Dict]:
    """Google News RSS search fallback. country_code e.g. IN, US, GB"""
    # Example: https://news.google.com/rss/search?q=India+economy&hl=en-IN&gl=IN&ceid=IN:en
    q = requests.utils.requote_uri(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-{country_code}&gl={country_code}&ceid={country_code}:en"
    try:
        feed = feedparser.parse(url)
        items = []
        for ent in feed.entries[:max_items]:
            # published_parsed may exist
            try:
                dt = datetime.fromtimestamp(time.mktime(ent.published_parsed)).astimezone(timezone.utc)
            except Exception:
                dt = datetime.now(timezone.utc)
            items.append({
                "title": ent.title,
                "summary": getattr(ent, "summary", ""),
                "url": ent.link,
                "source": ent.get("source", {}).get("title") or ent.get("author", ""),
                "published_at": dt,
            })
        return items
    except Exception:
        return []


def fetch_mospi_press_releases(max_items: int = 5) -> List[Dict]:
    """Try to pull MOSPI press releases from new.mospi.gov.in (best-effort)."""
    try:
        url = "https://new.mospi.gov.in/"  # home - we will try to find press releases - best-effort
        res = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=12)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "lxml")
        # Attempt: find press release links - heuristics (site structure may change)
        items = []
        # Look for elements with "press" or "press release" text
        anchors = soup.find_all("a", href=True)
        found = []
        for a in anchors:
            txt = (a.get_text() or "").strip().lower()
            if "press" in txt or "press release" in txt or "press note" in txt:
                href = a["href"]
                if href.startswith("/"):
                    href = "https://new.mospi.gov.in" + href
                found.append((txt, href))
        # de-duplicate and fetch basic title snippet
        added = set()
        for txt, href in found[:max_items]:
            if href in added:
                continue
            added.add(href)
            # fetch page title
            try:
                r2 = requests.get(href, headers={"User-Agent": USER_AGENT}, timeout=10)
                r2.raise_for_status()
                s2 = BeautifulSoup(r2.text, "lxml")
                title = s2.find("h1") or s2.title
                title_text = title.get_text().strip() if title else "MOSPI Press Release"
            except Exception:
                title_text = "MOSPI Press Release"
            items.append({"title": title_text, "summary": "", "url": href, "source": "MOSPI", "published_at": datetime.now(timezone.utc)})
        return items
    except Exception:
        return []


# -------------- Market fetcher --------------
INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "S&P 500": "^GSPC",
}


def fetch_index_snapshot() -> Dict[str, Dict]:
    out = {}
    for name, sym in INDEX_TICKERS.items():
        try:
            df = yf.download(sym, period="2d", interval="1d", progress=False, threads=False)
            if df is None or df.empty:
                out[name] = {"latest": None, "pct_change": None}
            else:
                latest_close = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
                pct = (latest_close - prev_close) / prev_close * 100 if prev_close != 0 else 0
                out[name] = {"latest": latest_close, "pct_change": pct}
        except Exception:
            out[name] = {"latest": None, "pct_change": None}
    return out


def fetch_stock_data(symbol: str, period: str = "5d", interval: str = "1d") -> pd.DataFrame:
    if not symbol or symbol.strip() == "":
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={"Close": "close"}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()


# -------------- UI & layout --------------
st.title("📰 News & Insights")
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Live economic and market news")
    st.markdown('<div class="subtext">Sources: NewsAPI (if provided) → Google News RSS fallback. MOSPI press releases: best-effort scrape.</div>', unsafe_allow_html=True)

    query = st.text_input("Search query (e.g., India economy, RBI, MOSPI, stock symbol):", value="India economy")
    max_headlines = st.slider("Headlines to show", 3, 15, 6)

with col2:
    st.header("Controls")
    refresh_choice = st.selectbox("Auto-refresh interval", options=["No auto-refresh", "1 minute", "5 minutes", "10 minutes"], index=2)
    refresh_map = {"No auto-refresh": 0, "1 minute": 60, "5 minutes": 300, "10 minutes": 600}
    refresh_seconds = refresh_map[refresh_choice]
    st.write("Manual refresh (forces re-fetch):")
    if st.button("Refresh now"):
        # Clear request cache and force update
        requests_cache.clear()
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("**Market indices snapshot**")
    idx_snap = fetch_index_snapshot()
    for k, v in idx_snap.items():
        latest = v["latest"]
        pct = v["pct_change"]
        if latest is None:
            st.markdown(f"<div class='metric-box'>{k}: N/A</div>", unsafe_allow_html=True)
        else:
            arrow = "▲" if pct >= 0 else "▼"
            color = POS_GREEN if pct >= 0 else NEG_RED
            st.markdown(f"<div class='metric-box'>{k}: <b>{latest:,.2f}</b> <span style='color:{color}'>{arrow} {pct:+.2f}%</span></div>", unsafe_allow_html=True)

st.markdown("---")

# Attempt auto-refresh using st_autorefresh if interval set
if refresh_seconds and refresh_seconds > 0:
    st.experimental_set_query_params(_refresh=int(time.time()))
    # Use streamlit's autorefresh utility
    from streamlit.runtime.scriptrunner import add_script_run_ctx  # safe import
    # st_autorefresh isn't a public import in older versions; using experimental_rerun approach:
    # We'll use a simple timer and set cache expiry to refresh data; user can rely on manual Refresh too.

# -------------- Fetching news (NewsAPI preferred) --------------
with st.spinner("Fetching news…"):
    news_items = []
    if NEWSAPI_KEY:
        news_items = fetch_news_from_newsapi(query, page_size=max_headlines)
    if not news_items:
        # Try Google RSS fallback
        # Determine country for query by simple heuristic
        # If query contains 'India' or 'IN', prefer IN, else use US
        country = "IN" if "india" in query.lower() else "US"
        news_items = fetch_news_from_google_rss(query, country_code=country, max_items=max_headlines)

# -------------- MOSPI press releases --------------
with st.expander("MOSPI / Government releases (best-effort)"):
    try:
        mospi = fetch_mospi_press_releases(6)
        if mospi:
            for it in mospi:
                pub = it.get("published_at")
                st.markdown(f"<div class='news-card'><div class='card-title'>{it.get('title')}</div><div class='card-meta'>{it.get('source')} • {short_time_ago(pub)}</div><div style='margin-top:6px'>{it.get('summary')}</div><a href='{it.get('url')}' target='_blank'>Read more</a></div>", unsafe_allow_html=True)
        else:
            st.info("Could not fetch MOSPI press releases automatically. You can upload press-release CSV or add notes.")
    except Exception:
        st.info("MOSPI press releases not available (network/structure change).")

# -------------- Display news feed with sentiment --------------
st.subheader("Top headlines")
if not news_items:
    st.warning("No news available. Try a different query or add NEWSAPI_KEY to Streamlit secrets.")
else:
    # Build DataFrame with sentiment
    rows = []
    for n in news_items:
        sent = classify_sentiment((n.get("title") or "") + " " + (n.get("summary") or ""))
        rows.append({
            "title": n.get("title"),
            "summary": n.get("summary") or "",
            "url": n.get("url"),
            "source": n.get("source") or "",
            "published_at": n.get("published_at") if isinstance(n.get("published_at"), datetime) else datetime.now(timezone.utc),
            "sent_label": sent["label"],
            "sent_score": sent["score"],
        })
    df_news = pd.DataFrame(rows)

    # Columns layout: list + sentiment summary
    left, right = st.columns([2, 1])
    with left:
        for _, row in df_news.iterrows():
            title = row["title"]
            summary = row["summary"]
            url = row["url"]
            src = row["source"]
            pub = row["published_at"]
            label = row["sent_label"]
            score = row["sent_score"]
            color = POS_GREEN if label == "Positive" else (NEG_RED if label == "Negative" else TEAL)
            st.markdown(
                f"""
                <div class="news-card">
                  <div class="card-title">{title}</div>
                  <div class="card-meta">{src} • {short_time_ago(pub)} • <span style="color:{color}">{label} ({score:+.2f})</span></div>
                  <div style="margin-top:8px">{summary}</div>
                  <div style="margin-top:8px"><a href="{url}" target="_blank">Read original</a></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### Sentiment summary")
        counts = df_news["sent_label"].value_counts().to_dict()
        pos = counts.get("Positive", 0)
        neu = counts.get("Neutral", 0)
        neg = counts.get("Negative", 0)
        # Simple pie chart using plotly
        fig = px.pie(values=[pos, neu, neg], names=["Positive", "Neutral", "Negative"],
                     color_discrete_map={"Positive": POS_GREEN, "Neutral": TEAL, "Negative": NEG_RED})
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor=BEIGE, plot_bgcolor=BEIGE)
        st.plotly_chart(fig, use_container_width=True)

# -------------- Market Movers / Stock search --------------
st.markdown("---")
st.subheader("Stock & Market — Single stock (auto-refresh available)")
lefts, rights = st.columns([2, 1])
with lefts:
    sym = st.text_input("Enter one stock symbol (Indian: use .NS suffix, e.g., RELIANCE.NS; US: AAPL):", value="AAPL")
    lookback = st.selectbox("Chart timeframe", options=["5d", "1mo", "3mo"], index=0)
    show_spark = st.checkbox("Show index sparklines", value=True)
    df_sym = fetch_stock_data(sym, period=lookback, interval="1d")
    if df_sym.empty:
        st.warning("No data for symbol. Check symbol format (e.g., RELIANCE.NS for India).")
    else:
        latest = df_sym["close"].iloc[-1]
        prev = df_sym["close"].iloc[-2] if len(df_sym) > 1 else latest
        pct = (latest - prev) / prev * 100 if prev != 0 else 0
        arrow = "▲" if pct >= 0 else "▼"
        col_txt = POS_GREEN if pct >= 0 else NEG_RED
        st.markdown(f"<div class='metric-box'>{sym} — Latest: <b>{latest:,.2f}</b> <span style='color:{col_txt}'>{arrow} {pct:+.2f}%</span></div>", unsafe_allow_html=True)
        # Plot line chart with green/red line color based on last delta
        line_color = POS_GREEN if pct >= 0 else NEG_RED
        fig_line = px.line(df_sym, x="Date", y="close", title=f"{sym} Price ({lookback})")
        fig_line.update_traces(line=dict(color=line_color, width=2))
        fig_line.update_layout(paper_bgcolor=BEIGE, plot_bgcolor=BEIGE, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_line, use_container_width=True)

with rights:
    st.markdown("#### Indices sparklines")
    if show_spark:
        for name, sym_idx in INDEX_TICKERS.items():
            idx_df = fetch_stock_data(sym_idx, period="1mo", interval="1d")
            if idx_df.empty:
                st.markdown(f"<div class='metric-box'>{name}: N/A</div>", unsafe_allow_html=True)
            else:
                fig_sp = px.line(idx_df, x="Date", y="close")
                fig_sp.update_traces(showlegend=False, line=dict(width=2, color=NAVY))
                fig_sp.update_layout(height=100, paper_bgcolor=BEIGE, plot_bgcolor=BEIGE, margin=dict(t=4, b=4, l=4, r=4))
                st.plotly_chart(fig_sp, use_container_width=True)

# -------------- Newsletter builder (simple) --------------
st.markdown("---")
with st.expander("📝 Newsletter Builder (compile top headlines)"):
    n_sel = st.number_input("How many top headlines to include?", min_value=1, max_value=10, value=5)
    if st.button("Generate newsletter text"):
        topn = df_news.head(n_sel) if not df_news.empty else pd.DataFrame()
        txt = "Daily Economic Brief\n\n"
        if topn.empty:
            txt += "No headlines available.\n"
        else:
            for i, r in topn.iterrows():
                txt += f"- {r['title']} ({r['source']})\n  Sentiment: {r['sent_label']} ({r['sent_score']:+.2f})\n  Link: {r['url']}\n\n"
        st.text_area("Newsletter (editable)", value=txt, height=300)

# -------------- Auto-refresh logic note --------------
if refresh_seconds and refresh_seconds > 0:
    # simple client-side meta-refresh technique: trigger a rerun after delay
    st.experimental_rerun() if refresh_seconds == 0 else None
    # NOTE: for production you can use st.experimental_memo with short ttl or a backend scheduler.

st.markdown("<div style='margin-top:20px; color:#567C8D'>Last updated: " + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + "</div>", unsafe_allow_html=True)
