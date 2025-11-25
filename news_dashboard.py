"""
news_dashboard.py
India Economic Intelligence — Full production-ready app (detailed)

Features:
- Color theme: SkyBlue -> Beige gradient, Navy headings, Teal subtext, white cards
- Shimmer loading while fetching news/indices
- Auto-refresh (streamlit-autorefresh if installed) with safe fallback
- News: NewsAPI (if NEWSAPI_KEY env) OR Google News RSS fallback
- Sentiment: TextBlob (inline badge)
- Personalized feed: interests + click history
- Market indices snapshot via yfinance
- Per-stock deep view (1y chart, price metric, dividends/splits, related news)
- MOSPI / data.gov integration for CPI / IIP / GDP micro-charts (auto) + CSV upload fallback
- Auto newsletter (3-4 bullets), editable, downloadable, optional SMTP send
- Caching for network calls, debug log in UI

Run:
1. pip install -r requirements.txt
2. (optional) create .env with NEWSAPI_KEY, DATA_GOV_API_KEY, CPI_RESOURCE_ID, IIP_RESOURCE_ID, GDP_RESOURCE_ID, SMTP_*
3. streamlit run news_dashboard.py
"""

import os
import time
import textwrap
from datetime import datetime
from collections import Counter, defaultdict
from io import BytesIO

import requests
import feedparser
import requests_cache
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob

import streamlit as st

# ---- Number animation helpers ----
def format_price(value):
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "N/A"


def build_index_card_html(name, price, pct):
    """Return HTML for the blue index card, for a given price + % change."""
    if pct is None or price is None:
        body = "N/A"
        change_str = ""
        color = PALETTE["neu"]
        arrow = ""
    else:
        color = PALETTE["pos"] if pct >= 0 else PALETTE["neg"]
        arrow = "▲" if pct >= 0 else "▼"
        body = format_price(price)
        change_str = f"{arrow} {pct:+.2f}%"

    return f"""
    <div class='card' style="text-align:left; padding:16px;">
        <div style="font-weight:700; font-size:14px; color:{PALETTE['navy']};">{name}</div>
        <div style="font-size:20px; margin-top:6px;">{body}</div>
        <div style="font-size:13px; font-weight:600; color:{color}; margin-top:4px;">
            {change_str}
        </div>
    </div>
    """


def animate_index_card(name, val, state_key):
    """
    Animate index card number from previous to current on each rerun.
    - name: 'NIFTY 50'
    - val: dict with 'last' and 'pct'
    - state_key: key for st.session_state to store previous last price
    """
    placeholder = st.empty()

    current = val.get("last")
    pct = val.get("pct")
    if current is None:
        # just draw once if no data
        placeholder.markdown(build_index_card_html(name, None, None), unsafe_allow_html=True)
        return

    current = float(current)

    # previous value from last run (for animation start)
    prev = st.session_state.get(state_key, current)

    # simple linear interpolation (20 frames)
    steps = 20
    for p in np.linspace(prev, current, steps):
        html = build_index_card_html(name, p, pct)
        placeholder.markdown(html, unsafe_allow_html=True)
        time.sleep(0.02)  # 20 ms per frame → quick smooth effect

    # store last value for next rerun
    st.session_state[state_key] = current


def animate_metric(label, value, delta, state_key):
    """
    Animate a Streamlit metric (for single stock price).
    - label: text label (e.g. 'Price')
    - value: current numeric value
    - delta: text for the delta (e.g. '+12.30')
    """
    box = st.empty()
    if value is None:
        box.metric(label, "N/A", delta)
        return

    current = float(value)
    prev = st.session_state.get(state_key, current)

    steps = 20
    for v in np.linspace(prev, current, steps):
        box.metric(label, f"₹{v:,.2f}", delta)
        time.sleep(0.02)

    st.session_state[state_key] = current
    
# --- Function to fetch fresh news ---
import requests
import pandas as pd
import pandas as pd
import PyPDF2
import io

def load_uploaded_df(uploaded_file):
    """Handles CSV, XLSX, and PDF uploads and returns a dataframe or text."""
    if uploaded_file is None:
        return None

    filename = uploaded_file.name.lower()

    # --- CSV File ---
    if filename.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None

    # --- Excel File ---
    elif filename.endswith(".xlsx"):
        try:
            return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel: {e}")
            return None

    # --- PDF File ---
    elif filename.endswith(".pdf"):
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() + "\n"
            return pdf_text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None

    else:
        st.warning("Unsupported file format. Please upload CSV, XLSX, or PDF.")
        return None
st.markdown("""
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 0rem;
    }
    .stDataFrame {
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

def fetch_latest_news(topic):
    """Fetch real-time news from GNews API (last ~15 minutes)."""
    url = f"https://gnews.io/api/v4/search?q={topic}&lang=en&country=in&max=10&sortby=publishedAt&token={st.secrets['GNEWS_API_KEY']}"
    try:
        data = requests.get(url).json()
        df = pd.DataFrame(data["articles"])[["title", "publishedAt", "url"]]
        return df
    except Exception as e:
        st.warning(f"Could not fetch latest news: {e}")
        return pd.DataFrame()

# optional autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREF = True
except Exception:
    HAS_AUTOREF = False

# dotenv (optional) - if you put keys in .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ---------- CONFIG / KEYS ----------
# Palette
PALETTE = {
    "sky": "#C8D9E6",
    "beige": "#F5EFEB",
    "navy": "#2F4156",
    "teal": "#567C8D",
    "white": "#FFFFFF",
    "pos": "#00C49F",
    "neg": "#FF4C4C",
    "neu": "#F5B041",
    "card": "#FFFFFF"
}

# API keys & resource ids from env or Streamlit secrets
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "").strip()
CPI_RESOURCE_ID = os.getenv("CPI_RESOURCE_ID", "").strip()
IIP_RESOURCE_ID = os.getenv("IIP_RESOURCE_ID", "").strip()
GDP_RESOURCE_ID = os.getenv("GDP_RESOURCE_ID", "").strip()

# SMTP settings (optional) for sending newsletter
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT")) if os.getenv("SMTP_PORT") else None
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()

# Indices
INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "S&P 500": "^GSPC",
}

# caching HTTP requests to reduce repeated calls
requests_cache.install_cache("news_cache", expire_after=180)

# caching TTLs for streamlit
NEWS_TTL = 120
MARKET_TTL = 10   # was 60 → now refresh every ~10 seconds
MACRO_TTL = 1800

# debug log
if "_log" not in st.session_state:
    st.session_state["_log"] = []

def log(msg):
    st.session_state["_log"].append(f"{datetime.utcnow().isoformat()}  {msg}")

# ---------- PAGE & CSS ----------
st.set_page_config(page_title="India Economic Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
:root {{
  --sky: {PALETTE['sky']};
  --beige: {PALETTE['beige']};
  --navy: {PALETTE['navy']};
  --teal: {PALETTE['teal']};
  --white: {PALETTE['white']};
  --pos: {PALETTE['pos']};
  --neg: {PALETTE['neg']};
  --neu: {PALETTE['neu']};
}}
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, var(--sky) 0%, var(--beige) 100%);
}}
h1, h2, h3, h4 {{ color: var(--navy); font-weight:700; }}
.card {{
  background: var(--white);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}}
.small-muted {{ color: var(--teal); font-size:0.95em; }}
.sent-badge {{ display:inline-block; padding:4px 8px; border-radius:10px; color:white; font-weight:600; font-size:12px; }}
.skel {{ height:16px; border-radius:8px; background: linear-gradient(90deg, var(--sky), var(--white), var(--sky)); background-size:200% 100%; animation: shimmer 1.2s infinite linear; }}
@keyframes shimmer {{ from {{background-position:-200% 0}} to {{background-position:200% 0}} }}
.block-container {{ padding-top:1rem; padding-bottom:1rem; }}
</style>
""", unsafe_allow_html=True)

# ---------- UTILITIES ----------
def safe_json_get(url, params=None, headers=None, timeout=12):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"HTTP error: {e} | url={url}")
        return None

# NewsAPI fetch (cached)
@st.cache_data(ttl=NEWS_TTL)
def fetch_newsapi(query, n=10):
    if not NEWSAPI_KEY:
        return None

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": n,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }

    try:
        js = safe_json_get(url, params=params)
        if js and js.get("status") == "ok":
            out = []
            for a in js.get("articles", [])[:n]:
                out.append(
                    {
                        "title": a.get("title"),
                        "summary": a.get("description") or a.get("content") or "",
                        "url": a.get("url"),
                        "source": (a.get("source") or {}).get("name"),
                        "publishedAt": a.get("publishedAt"),
                    }
                )
            return out
    except Exception as e:
        log(f"newsapi error: {e}")
        return None


@st.cache_data(ttl=NEWS_TTL)
def fetch_google_rss(query, n=10, country="IN"):
    q = requests.utils.requote_uri(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-{country}&gl={country}&ceid={country}:en"
    try:
        feed = feedparser.parse(url)
        out = []
        for entry in feed.entries[:n]:
            out.append(
                {
                    "title": entry.get("title"),
                    "summary": entry.get("summary") or "",
                    "url": entry.get("link"),
                    "source": (entry.get("source") or {}).get("title")
                    if entry.get("source")
                    else None,
                    "publishedAt": entry.get("published") or entry.get("published_parsed"),
                }
            )
        return out
    except Exception as e:
        log(f"google rss error: {e}")
        return []


# -------- Unified news fetch with TODAY filter (IST) --------
def fetch_news(query, n=8, only_today=False):
    """
    Fetch news via NewsAPI or Google RSS fallback.
    - only_today=True → keeps only IST-date articles
    """

    def _parse_pub_to_utc(pub):
        """Convert raw publishedAt to tz-aware UTC timestamp."""
        try:
            ts = pd.to_datetime(pub, utc=True, errors="coerce")
            if pd.isna(ts):
                return None
            return ts          # tz-aware UTC
        except Exception:
            return None

    # Choose source: NewsAPI if we have a key, otherwise Google RSS
    res = fetch_newsapi(query, n=n) if NEWSAPI_KEY else None
    raw = res if res else fetch_google_rss(query, n=n)

    if not raw:
        return []
        
# --- normalise records ---
    cleaned = []
    for a in raw[:n]:
        item = {
            "title": a.get("title") or "",
            "summary": a.get("summary") or a.get("description") or "",
            "url": a.get("url") or a.get("link") or "",
            "source": (a.get("source") or {}).get("name")
                      if isinstance(a.get("source"), dict) else a.get("source"),
            "publishedAt_raw": a.get("publishedAt")
                               or a.get("published")
                               or a.get("pubDate") or "",
        }
        item["publishedAt"] = _parse_pub_to_utc(item["publishedAt_raw"])
        cleaned.append(item)

    # Filter ONLY TODAY’S articles in IST
    if only_today:
        now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
        today_ist = now_ist.date()

        filtered = []
        for it in cleaned:
            ts = it.get("publishedAt")
            if ts is None:
                continue
            ts_ist = ts.tz_convert("Asia/Kolkata")
            if ts_ist.date() == today_ist:
                filtered.append(it)
        cleaned = filtered

    return cleaned
   
def sentiment_label(text):
    try:
        tb = TextBlob(text or "")
        score = round(tb.sentiment.polarity, 3)
        if score >= 0.05:
            return ("positive", score)
        elif score <= -0.05:
            return ("negative", score)
        else:
            return ("neutral", score)
    except Exception as e:
        log(f"sentiment error: {e}")
        return ("neutral", 0.0)
        
# yfinance helpers
@st.cache_data(ttl=MARKET_TTL)
def fetch_index_snapshot():
    """
    Live-ish snapshot for major indices.

    - Uses intraday 5-minute candles (last 2 days) from Yahoo Finance.
    - 'last' = latest close
    - 'pct'  = % change vs previous 5-minute bar
    """
    out = {}
    for name, sym in INDICES.items():
        try:
            # 2 days + 5-minute interval gives intraday movement
            df = yf.download(
                sym,
                period="2d",
                interval="5m",
                progress=False,
                threads=False,
            )

            if df is None or df.empty:
                out[name] = {"last": None, "pct": None}
                continue

            # remove any duplicate timestamps just in case
            df = df[~df.index.duplicated(keep="last")]

            last = float(df["Close"].iloc[-1])
            prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last

            pct = (last - prev) / prev * 100 if prev != 0 else 0.0
            out[name] = {"last": last, "pct": pct}

        except Exception as e:
            out[name] = {"last": None, "pct": None}
            log(f"index fetch error {name}: {e}")

    return out
    
@st.cache_data(ttl=MARKET_TTL)
def fetch_stock_history(sym, period="1y", interval="1d"):
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"Close":"close"})
        return df
    except Exception as e:
        log(f"stock history error {sym}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=MARKET_TTL)
def fetch_stock_actions(sym):
    """
    Returns dividends, splits, news & upcoming events for a stock.
    """
    try:
        t = yf.Ticker(sym)

        # past cash actions
        divs = t.dividends if hasattr(t, "dividends") else pd.Series(dtype=float)
        splits = t.splits if hasattr(t, "splits") else pd.Series(dtype=float)

        # news
        news = []
        try:
            raw = getattr(t, "news", None)
            if isinstance(raw, list):
                for item in raw[:20]:
                    news.append(
                        {
                            "title": item.get("title"),
                            "link": item.get("link"),
                            "publisher": item.get("publisher"),
                            "providerPublishTime": item.get("providerPublishTime"),
                        }
                    )
        except Exception:
            pass

        # upcoming events (earnings, meetings etc.)
        events = []
        try:
            cal = getattr(t, "calendar", None)
            if cal is not None and not cal.empty:
                for idx, row in cal.iterrows():
                    events.append(
                        {
                            "type": str(idx),
                            "date": row.iloc[0],
                            "detail": str(row.iloc[0]),
                        }
                    )
        except Exception:
            pass

        # earnings dates (future)
        try:
            ed = getattr(t, "earnings_dates", None)
            if ed is not None and not ed.empty:
                ed = ed.reset_index()
                for _, r in ed.tail(4).iterrows():
                    events.append(
                        {
                            "type": "EARNINGS",
                            "date": r.iloc[0],
                            "detail": f"EPS: {r.get('EPS actual', '')} vs {r.get('EPS estimate', '')}",
                        }
                    )
        except Exception:
            pass

        return {
            "dividends": divs,
            "splits": splits,
            "news": news,
            "events": events,
        }
    except Exception as e:
        log(f"stock actions error {sym}: {e}")
        return {
            "dividends": pd.Series(dtype=float),
            "splits": pd.Series(dtype=float),
            "news": [],
            "events": [],
        }
        
# data.gov fetch
@st.cache_data(ttl=MACRO_TTL)
def fetch_data_gov_resource(resource_id, limit=1000, api_key=None):
    if not resource_id:
        return None
    key = api_key or DATA_GOV_API_KEY
    if not key:
        return None
    try:
        url = f"https://api.data.gov.in/resource/{resource_id}.json"
        params = {"api-key": key, "limit": limit}
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"data.gov fetch error {resource_id}: {e}")
        return None

def extract_trending_terms(headlines, top_n=8):
    stop = set(["the","and","for","with","from","that","this","are","was","will","have","has","india","govt","government"])
    words = []
    for h in headlines:
        if not h: continue
        for w in h.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if len(w) > 3 and w not in stop:
                words.append(w)
    return [w for w,_ in Counter(words).most_common(top_n)]

# personalization init
def init_personalization():
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = ["rbi","infrastructure","inflation"]
    if "click_counts" not in st.session_state:
        st.session_state["click_counts"] = defaultdict(int)

def record_click(aid):
    st.session_state["click_counts"][aid] += 1

def score_for_user(article, interests, trending):
    text = (article.get("title","") + " " + (article.get("summary") or "")).lower()
    score = 0
    for it in interests:
        if it.lower() in text:
            score += 2
    for t in trending:
        if t in text:
            score += 1
    aid = article.get("url") or article.get("title")
    score += st.session_state["click_counts"].get(aid, 0) * 0.5
    return score

# newsletter generation
def build_newsletter(top_articles, macro_bullets=[]):
    bullets = []
    if macro_bullets:
        bullets.append(macro_bullets[0])
    for a in top_articles[:3]:
        bullets.append(textwrap.shorten(a.get("title",""), width=140, placeholder="..."))
    if not bullets:
        bullets = ["No items available."]
    text = "Daily Economic Brief — Auto-generated\n\n" + "\n".join(f"{i+1}. {b}" for i,b in enumerate(bullets))
    return text

# small datetime formatter
def fmt_dt(val):
    if not val:
        return ""
    try:
        return pd.to_datetime(val).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(val)

# ---------- UI: Sidebar controls ----------
init_personalization()

st.sidebar.title("Controls & Settings")

# how many headlines to show
headlines_count = st.sidebar.slider(
    "Headlines to show", min_value=3, max_value=20, value=6
)

# auto-refresh interval for live indices + news + stock data
auto_ref = st.sidebar.selectbox(
    "Auto-refresh",
    options=["Off", "1s", "30s", "1m", "5m"],
    index=3,  # default = 1m
)

# single stock symbol
stock_input = st.sidebar.text_input(
    "Single stock (one symbol)", value="RELIANCE.NS"
)

st.sidebar.markdown("---")

# --- Interests (for personalization) ---
st.sidebar.markdown("### 🧠 Interests for personalization")

interests = st.sidebar.multiselect(
    "Pick interests",
    [
        "RBI",
        "infrastructure",
        "startups",
        "banks",
        "inflation",
        "GDP",
        "employment",
        "policy",
        "stock",
    ],
    default=["inflation", "RBI"],
)

if st.sidebar.button("Save interests"):
    st.session_state["interests"] = interests

# manual hard refresh of cache + app
if st.sidebar.button("Refresh now"):
    requests_cache.clear()  # clear cached HTTP responses
    st.experimental_rerun()

# --- parse auto_ref seconds & enable streamlit_autorefresh if available ---
interval_map = {"Off": 0, "1s": 1, "30s": 30, "1m": 60, "5m": 300}
interval_seconds = interval_map.get(auto_ref, 0)

if HAS_AUTOREF and interval_seconds > 0:
    tick = st_autorefresh(
        interval=interval_seconds * 1000,
        key="autorefresh_counter",
    )
    st.sidebar.caption(f"Auto-refresh ticks: {tick}")
elif interval_seconds > 0:
    st.sidebar.info(
        "Auto-refresh set; install streamlit-autorefresh for automatic reloads."
    )
    
# ---------- Top header & indices ----------
st.markdown("<h1>📰 News & Insights — India Economic Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Live economic, policy, infrastructure, employment, markets & corporate news — personalized feed</div>", unsafe_allow_html=True)
st.markdown("---")

with st.spinner("Fetching market snapshot..."):
    indices = fetch_index_snapshot()

with st.spinner("Fetching market snapshot..."):
    indices = fetch_index_snapshot()

# indices tiles (animated)
cols = st.columns(len(INDICES))

for i, (name, sym) in enumerate(INDICES.items()):
    val = indices.get(name, {"last": None, "pct": None})
    with cols[i]:
        # animate card using previous value from session_state
        animate_index_card(name, val, state_key=f"idx_{name}")
        
st.markdown("---")

# -------- Fetch news ----------
raw_news = []

# Default topic before anything else
search_query = "India economy"    # <--- FIXED & moved up

with st.spinner("Fetching news..."):
    if search_query.strip():
        # main attempt
        raw_news = fetch_news(search_query, n=headlines_count, only_today=True)

        # fallback: use first word only if nothing came back
        if not raw_news and " " in search_query:
            raw_news = fetch_news(
                search_query.split(" ")[0],
                n=headlines_count,
                only_today=True,
            )
            
if not raw_news:
    st.info("No news found for this query (NewsAPI may be required, or try another keyword).")
    
# enrich news with sentiment & user score
for a in raw_news:
    text = (a.get("title","") or "") + ". " + (a.get("summary") or "")
    label, score = sentiment_label(text)
    a["sent_label"] = label
    a["sent_score"] = score

headlines_text = [a.get("title","") for a in raw_news]
trending = extract_trending_terms(headlines_text, top_n=8)

# scoring and sorting
for a in raw_news:
    a["_user_score"] = score_for_user(a, st.session_state.get("prefs", []), trending)

def parse_pub(a):
    p = a.get("publishedAt") or a.get("published") or ""
    try:
        return pd.to_datetime(p)
    except Exception:
        return pd.Timestamp.min

news_sorted = sorted(raw_news, key=lambda x: (x["_user_score"], parse_pub(x)), reverse=True)
# ---------- Overall Sentiment Meter ----------
if raw_news:
    avg_sent = float(np.mean([a.get("sent_score", 0.0) for a in raw_news]))
    
    if avg_sent >= 0.05:
        mood = "😊 Overall Mood: Positive"
        bar_color = PALETTE["pos"]
    elif avg_sent <= -0.05:
        mood = "😟 Overall Mood: Negative"
        bar_color = PALETTE["neg"]
    else:
        mood = "😐 Overall Mood: Neutral"
        bar_color = PALETTE["neu"]

    st.markdown(
        f"""
        <div class='card' style="margin-bottom:10px; border-left:6px solid {bar_color}; padding:10px;">
            <div style="font-size:13px; color:{PALETTE['teal']};">Sentiment Meter</div>
            <div style="font-size:15px; font-weight:600;">{mood} (avg score: {avg_sent:+.2f})</div>
        </div>
        """,
        unsafe_allow_html=True
    )
# ---------- Breaking News Ribbon ----------
if raw_news:
    top_headline = news_sorted[0]

    bt = top_headline.get("title", "")
    burl = top_headline.get("url", "")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, {PALETTE['navy']}, {PALETTE['teal']});
            color:white;
            padding:6px 12px;
            border-radius:8px;
            font-size:13px;
            margin-bottom:12px;
            white-space:nowrap;
            overflow:hidden;
            text-overflow:ellipsis;
        ">
            <strong>🔔 Breaking:</strong>
            <a href="{burl}" target="_blank" style="color:white; text-decoration:none;">
                {bt}
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Layout: main + side ----------
main, side = st.columns([3,1])

with main:
    st.markdown("<div style='font-weight:700; font-size:20px'>Top headlines</div>", unsafe_allow_html=True)
    if not news_sorted:
        st.write("No headlines.")
    for idx,a in enumerate(news_sorted, start=1):
        title = a.get("title","")
        summary = a.get("summary","")
        url = a.get("url","")
        src = a.get("source","")
        pub = a.get("publishedAt") or a.get("published") or ""
        label = a.get("sent_label")
        sscore = a.get("sent_score")
        color = PALETTE["pos"] if label=="positive" else (PALETTE["neg"] if label=="negative" else PALETTE["neu"])
        badge_html = f"<span class='sent-badge' style='background:{color}'>{label.upper()}</span>"
        st.markdown(f"""
            <div class='card'>
              <div style='display:flex; justify-content:space-between; align-items:center;'>
                <div style='flex:1'>
                  <a href="{url}" target='_blank' style='text-decoration:none; color:{PALETTE['navy']}; font-weight:700'>{idx}. {title}</a>
                  <div class='small-muted'>{src} · {fmt_dt(pub)}</div>
                </div>
                <div style='text-align:right; margin-left:12px;'>
                  {badge_html}
                  <div style='color:{PALETTE['teal']}; font-size:12px; margin-top:6px;'>Score: {sscore:+.2f}</div>
                </div>
              </div>
              <div style='margin-top:8px; color:#222'>{summary}</div>
              <div style='margin-top:8px'><a href="{url}" target='_blank'>Read full article →</a></div>
            </div>
        """, unsafe_allow_html=True)
        # mark read button
        if st.button("Mark read", key=f"mr_{idx}"):
            record_click(a.get("url") or title)
            st.experimental_rerun()

with side:
    st.markdown("<div style='font-weight:700; font-size:18px'>Trending</div>", unsafe_allow_html=True)
    if trending:
        for t in trending:
            st.markdown(f"- {t}")
    else:
        st.write("-")
    st.markdown("---")
    st.markdown("<div style='font-weight:700'>For you</div>", unsafe_allow_html=True)
    top_for_you = sorted(news_sorted, key=lambda x: x["_user_score"], reverse=True)[:4]
    if not top_for_you:
        st.write("No personalized picks.")
    else:
        for t in top_for_you:
            st.markdown(f"- <a href='{t.get('url')}' target='_blank'>{t.get('title')}</a>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("Quick filters")
    if st.button("Only Positive"):
        st.session_state["_filter"] = "positive"
        st.experimental_rerun()
    if st.button("Only Negative"):
        st.session_state["_filter"] = "negative"
        st.experimental_rerun()
    if st.button("Reset Filter"):
        st.session_state.pop("_filter", None)
        st.experimental_rerun()

# apply filter if set
flt = st.session_state.get("_filter")
if flt:
    news_sorted = [n for n in news_sorted if n.get("sent_label")==flt]

# ---------- Newsletter auto-summary ----------
st.markdown("---")
st.markdown("### 📝 Auto Newsletter — short brief (editable)")
macro_bullets = []
if CPI_RESOURCE_ID and DATA_GOV_API_KEY:
    j = fetch_data_gov_resource(CPI_RESOURCE_ID, limit=10)
    if j and j.get("records"):
        df_c = pd.DataFrame(j["records"])
        date_col = next((c for c in df_c.columns if "date" in c.lower() or "month" in c.lower()), None)
        val_col = next((c for c in df_c.columns if any(x in c.lower() for x in ["value","cpi","index"])), None)
        if date_col and val_col:
            latest = df_c.sort_values(date_col).iloc[-1]
            macro_bullets.append(f"CPI ({val_col}) = {latest[val_col]} ({latest[date_col]})")

top_for_newsletter = top_for_you[:3]
nl_text = build_newsletter(top_for_newsletter, macro_bullets)
nl_area = st.text_area("Newsletter (editable)", value=nl_text, height=220)
st.download_button("Download newsletter (TXT)", data=nl_area.encode("utf-8"), file_name="economic_brief.txt")

# --- Admin-Only Upload Section (hidden by default) ---
with st.expander("⚙️ Admin: Upload New Data Files", expanded=False):
    st.info("This section is only for admin uploads. Expand when you want to update data or press release files.")
    cpi_upload = st.file_uploader("Upload CPI CSV/PDF (fallback)", type=["csv", "pdf"])
    iip_upload = st.file_uploader("Upload IIP CSV/PDF (fallback)", type=["csv", "pdf"])
    gdp_upload = st.file_uploader("Upload GDP CSV/PDF (fallback)", type=["csv", "pdf"])
    unemp_upload = st.file_uploader("Upload Unemployment CSV/PDF (fallback)", type=["csv", "pdf"])
    
# --- Load uploaded files for each indicator (CPI, IIP, GDP, Unemployment) ---
cpi_df_up = load_uploaded_df(cpi_upload)
iip_df_up = load_uploaded_df(iip_upload)
gdp_df_up = load_uploaded_df(gdp_upload)

try:
    unemp_df_up = load_uploaded_df(unemp_upload)
except NameError:
    unemp_df_up = None

# optional send via SMTP
if SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS:
    st.markdown("Send newsletter via SMTP")
    to_addr = st.text_input("To (comma separated):", value="")

    if st.button("Send newsletter"):
        import smtplib, ssl
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = "Daily Economic Brief"
        msg["From"] = SMTP_USER
        msg["To"] = [a.strip() for a in to_addr.split(",") if a.strip()]
        msg.set_content(nl_area)

        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
            st.success("✅ Newsletter sent successfully!")

        except Exception as e:
            st.error(f"❌ Failed to send email: {e}")
else:
    st.info("SMTP not configured. To enable send, set SMTP_* env vars.")

# ---------- INDIA'S ECONOMIC INDICATORS — MAIN PAGE & DETAILED MACRO DASHBOARD ----------
st.markdown("---")
st.markdown("<h2>📊 INDIA'S ECONOMIC INDICATORS — MAIN PAGE</h2>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Click any card to open the detailed macro dashboard (CPI, IIP, GDP, Unemployment)</div>", unsafe_allow_html=True)
st.markdown("")

def _load_uploaded_df(uploaded):
    if not uploaded:
        return None
    try:
        if uploaded.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded)
        else:
            return pd.read_pdf(uploaded)
    except Exception as e:
        log(f"upload parse error: {e}")
        return None

cpi_df_up = _load_uploaded_df(cpi_upload)
iip_df_up = _load_uploaded_df(iip_upload)
gdp_df_up = _load_uploaded_df(gdp_upload)

# --- Fetch latest small-summaries from data.gov if available (safe) ---
cpi_data_gov = None
iip_data_gov = None
gdp_data_gov = None
try:
    if CPI_RESOURCE_ID and DATA_GOV_API_KEY:
        j = fetch_data_gov_resource(CPI_RESOURCE_ID, limit=10)
        if j and j.get("records"):
            cpi_data_gov = pd.DataFrame(j["records"])
    if IIP_RESOURCE_ID and DATA_GOV_API_KEY:
        j = fetch_data_gov_resource(IIP_RESOURCE_ID, limit=10)
        if j and j.get("records"):
            iip_data_gov = pd.DataFrame(j["records"])
    if GDP_RESOURCE_ID and DATA_GOV_API_KEY:
        j = fetch_data_gov_resource(GDP_RESOURCE_ID, limit=10)
        if j and j.get("records"):
            gdp_data_gov = pd.DataFrame(j["records"])
except Exception as e:
    log(f"macro fetch error: {e}")

# -------- Demo fallback data if API + uploads are empty --------
if cpi_data_gov is None and cpi_df_up is None:
    # 12 months of fake CPI inflation
    dates = pd.date_range("2024-01-01", periods=12, freq="M")
    cpi_data_gov = pd.DataFrame({
        "Month": dates,
        "CPI_Inflation_rate": [5.2, 5.0, 4.8, 4.5, 4.3, 4.1, 3.9, 4.0, 4.2, 4.4, 4.6, 4.7],
    })

if iip_data_gov is None and iip_df_up is None:
    # 12 months of fake IIP growth
    dates = pd.date_range("2024-01-01", periods=12, freq="M")
    iip_data_gov = pd.DataFrame({
        "Month": dates,
        "IIP_Growth_percent": [2.3, 3.1, 4.0, 3.8, 4.5, 5.1, 4.9, 5.3, 5.0, 4.7, 4.9, 5.2],
    })

if gdp_data_gov is None and gdp_df_up is None:
    # 8 quarters of fake GDP growth
    quarters = [
        "Q1 2023-24", "Q2 2023-24", "Q3 2023-24", "Q4 2023-24",
        "Q1 2024-25", "Q2 2024-25", "Q3 2024-25", "Q4 2024-25",
    ]
    gdp_data_gov = pd.DataFrame({
        "Quarter": quarters,
        "Real_GDP_growth_percent": [7.8, 7.2, 8.1, 7.6, 7.9, 8.2, 7.7, 8.0],
    })

# --- Small helper: get latest numeric summary from df-like object ---
def latest_summary_from_df(df, date_cols=None, value_cols=None):
    if df is None or df.empty:
        return None, None
    try:
        # try to auto-detect date and value columns
        cols = df.columns.tolist()
        date_col = None
        val_col = None
        for c in cols:
            if "date" in c.lower() or "month" in c.lower() or "quarter" in c.lower():
                date_col = c; break
        for c in cols:
            if any(x in c.lower() for x in ["value","index","cpi","gdp","iip","amount","growth","percent","%","rate"]):
                val_col = c; break
        if date_col is None:
            date_col = cols[0]
        if val_col is None and len(cols) > 1:
            val_col = cols[1]
        tmp = df.copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp = tmp.dropna(subset=[date_col])
        latest_row = tmp.sort_values(date_col).iloc[-1]
        return latest_row[val_col], latest_row[date_col]
    except Exception as e:
        log(f"latest_summary error: {e}")
        return None, None

# Pick source df: API data if present, otherwise uploaded CSV, otherwise None
cpi_source = cpi_data_gov if cpi_data_gov is not None else cpi_df_up
iip_source = iip_data_gov if iip_data_gov is not None else iip_df_up
gdp_source = gdp_data_gov if gdp_data_gov is not None else gdp_df_up

cpi_val, cpi_date = latest_summary_from_df(cpi_source)
iip_val, iip_date = latest_summary_from_df(iip_source)
gdp_val, gdp_date = latest_summary_from_df(gdp_source)
# Unemployment – currently only from upload (no data.gov API)
if isinstance(unemp_df_up, pd.DataFrame):
    unemp_val, unemp_date = latest_summary_from_df(unemp_df_up)
else:
    unemp_val, unemp_date = None, None
    
# --- Overview Cards styled like National Statistics Office ---
st.markdown("<h3 style='margin-top:15px;'>Key Indicators (Click any to explore full dashboard)</h3>", unsafe_allow_html=True)

cards = [
    {"label": "Index of Industrial Production", "short": "IIP", "icon": "🏭", "key": "iip",
     "val": iip_val or 4.0, "date": str(iip_date.date()) if iip_date is not None else "September 2025"},
    {"label": "Inflation Rate (CPI Based)", "short": "CPI", "icon": "📊", "key": "cpi",
     "val": cpi_val or 1.54, "date": str(cpi_date.date()) if cpi_date is not None else "September 2025"},
    {"label": "Gross Domestic Product (Growth)", "short": "GDP", "icon": "💹", "key": "gdp",
     "val": gdp_val or 7.8, "date": str(gdp_date.date()) if gdp_date is not None else "Q1 2025–26"},
    {"label": "Unemployment Rate", "short": "UNEMP", "icon": "👷", "key": "unemp",
     "val": unemp_val or 5.2, "date": str(unemp_date.date()) if unemp_date is not None else "September 2025"},
]

cols = st.columns(4, gap="large")

for col, card in zip(cols, cards):
    icon = card["icon"]
    label = card["label"]
    val = card["val"]
    date_text = card["date"]
    key = card["key"]

    # Convert value to formatted string with % sign
    value_display = f"{val:.1f}%" if isinstance(val, (int,float)) else "N/A"

    # HTML card block
    html_card = f"""
    <div style='
        background: linear-gradient(180deg, #052e6f 0%, #021c47 100%);
        color: white;
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 0px 12px rgba(255,255,255,0.1);
        transition: transform 0.2s ease-in-out;
    ' onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1.00)'">
        <div style='font-size: 48px; margin-bottom: 8px;'>{icon}</div>
        <div style='font-size: 32px; color: #FFA500; font-weight: 700;'>{value_display}</div>
        <div style='font-size: 16px; font-weight: 600; margin-top: 5px;'>{label}</div>
        <div style='font-size: 13px; color: #ccc; margin-top: 3px;'>{date_text}</div>
    </div>
    """

    # Show the card
    col.markdown(html_card, unsafe_allow_html=True)

    # Clear, visible button to open the panel
    if col.button(f"Open {card['short']} details", key=f"btn_{key}"):
        st.session_state["macro_panel"] = key

st.markdown("---")

# initialize navigation state
if "macro_panel" not in st.session_state:
    st.session_state["macro_panel"] = None

# --- Helper: show press releases and news with sentiment for a keyword ---
def read_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()[:2000]  # Limit to 2000 chars for display
    except Exception as e:
        return f"⚠️ Could not read PDF: {e}"
def show_press_and_news(keyword, resource_id=None, uploaded_df=None, nnews=6):
    """
    Show press releases (official) and related news (sentiment-labeled)
    for a given keyword. Accepts an optional data.gov resource_id or an
    uploaded dataframe (uploaded_df) as fallback.
    """
# --- PRESS RELEASE SECTION ---
    st.markdown("### ⚖️ Press releases / Latest official data")

    if resource_id and DATA_GOV_API_KEY:
        try:
            j = fetch_data_gov_resource(resource_id, limit=6)
        except Exception as e:
            st.warning(f"⚠️ Unable to fetch data.gov resource: {e}")
            j = None

        if j and j.get("records"):
            for rec in j["records"][:6]:
                title = rec.get("title") or rec.get("indicator") or rec.get("month") or "Release"
                st.markdown(f"- **{title}** · {list(rec.items())[:1]}")
        else:
            st.info("No official recent releases found (data.gov).")

    elif uploaded_df is not None:
        try:
            # Handle PDF or CSV/XLSX
            if hasattr(uploaded_df, "name") and uploaded_df.name.lower().endswith(".pdf"):
                pdf_text = read_pdf(uploaded_df)
                st.markdown("##### 📰 Extracted press release preview:")
                st.text_area("Press Release Content", pdf_text, height=200)
            else:
                st.dataframe(uploaded_df.head(6))
        except Exception as e:
            st.warning(f"⚠️ Error reading uploaded file: {e}")

    else:
        st.info("No official release data available. Upload CSV/PDF as fallback.")
        
    # --- NEWS SECTION ---
    st.markdown("#### 🗞️ Related news (sentiment-labeled)")

    try:
        related = fetch_news(keyword, n=nnews)
        if not related:
            st.info("No news found.")
            return

        for a in related:
            t = a.get("title") or a.get("headline") or ""
            s = a.get("summary") or a.get("description") or ""
            label, score = sentiment_label((t or "") + " " + (s or ""))
            color = PALETTE["pos"] if label == "positive" else (
                PALETTE["neg"] if label == "negative" else PALETTE["neu"]
            )

            st.markdown(
                f"📰 **[{t}]({a.get('url')})** — "
                f"<span style='color:{color}; font-weight:700'>{label.upper()}</span> ({score:+.2f})",
                unsafe_allow_html=True,
            )

            if s and len(s) < 300:
                st.caption(s)

    except Exception as e:
        st.warning(f"⚠️ Unable to fetch or display related news: {e}")
        return
        
# --- Detailed dashboard panel renderer (shows four sections in tabs / collapsible) ---

            # ------- Year-on-Year change chart (if enough history) -------
            if len(tmp) >= 13:
                st.markdown("#### Approx. Year-on-Year change (%)")

                yoy_df = tmp.copy()
                # 12-period change → works reasonably for monthly data
                yoy_df["YoY_change_%"] = yoy_df[value_col].pct_change(periods=12) * 100
                yoy_df = yoy_df.dropna(subset=["YoY_change_%"])

                if not yoy_df.empty:
                    fig_yoy = px.line(
                        yoy_df,
                        x=date_col,
                        y="YoY_change_%"
                    )
                    fig_yoy.update_layout(
                        xaxis_title="Period",
                        yaxis_title="YoY change (%)",
                        height=320,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_yoy, use_container_width=True)
# ------- Smooth area chart (last 24 periods) -------
            st.markdown("#### Last 24 periods (smooth area)")
            area_df = tmp.tail(24)
            if not area_df.empty:
                fig_area = px.area(
                    area_df,
                    x=date_col,
                    y=value_col,
                )
                fig_area.update_layout(
                    xaxis_title="Period (recent)",
                    yaxis_title=value_col,
                    height=320,
                    template="plotly_white",
                )
                st.plotly_chart(fig_area, use_container_width=True)

            # ------- Year-on-Year change chart (if enough history) -------
            if len(tmp) >= 13:
                st.markdown("#### Approx. Year-on-Year change (%)")

                yoy_df = tmp.copy()
                # 12-period change → works well for monthly data
                yoy_df["YoY_change_%"] = yoy_df[value_col].pct_change(periods=12) * 100
                yoy_df = yoy_df.dropna(subset=["YoY_change_%"])

                if not yoy_df.empty:
                    fig_yoy = px.line(
                        yoy_df,
                        x=date_col,
                        y="YoY_change_%"
                    )
                    fig_yoy.update_layout(
                        xaxis_title="Period",
                        yaxis_title="YoY change (%)",
                        height=320,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_yoy, use_container_width=True)

            # ------- Distribution of values (histogram) -------
            st.markdown("#### Distribution of values")
            try:
                hist_df = tmp[[value_col]].dropna()
                if not hist_df.empty:
                    fig_hist = px.histogram(
                        hist_df,
                        x=value_col,
                        nbins=20,
                    )
                    fig_hist.update_layout(
                        xaxis_title=value_col,
                        yaxis_title="Frequency",
                        height=300,
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
            except Exception as e:
                st.caption(f"Histogram not available: {e}")
def render_macro_detail():
    panel = st.session_state.get("macro_panel")
    if not panel:
        return

    # header + back button
    st.button(
        "← Back to Overview",
        key="back_macro",
        on_click=lambda: st.session_state.update({"macro_panel": None})
    )
    st.markdown(
        f"<h3 style='margin-top:6px'>Detailed macro dashboard — {panel.upper()}</h3>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small-muted'>Data-driven visualizations, press releases and related news</div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # We always show the four sections, with the clicked one opened by default
    sections = ["gdp", "cpi", "iip", "unemp"]

    for sec in sections:
        with st.expander(sec.upper(), expanded=(sec == panel)):
            left, right = st.columns([2, 1])

            # ---------- pick data frame for this section ----------
            if sec == "cpi":
                df_try = cpi_data_gov if cpi_data_gov is not None else cpi_df_up
            elif sec == "iip":
                df_try = iip_data_gov if iip_data_gov is not None else iip_df_up
            elif sec == "gdp":
                df_try = gdp_data_gov if gdp_data_gov is not None else gdp_df_up
            else:  # unemployment – only from upload for now
                df_try = unemp_df_up
# --- If no real data, create a smooth demo series so charts still work ---
        if df_try is None:
            dates = pd.date_range("2024-01-01", periods=12, freq="M")

            if sec == "cpi":
                vals = np.linspace(6.5, 4.0, len(dates)) + np.random.normal(0, 0.15, len(dates))
                df_try = pd.DataFrame({"Date": dates, "CPI_Index": np.round(vals, 2)})

            elif sec == "iip":
                vals = np.linspace(110, 125, len(dates)) + np.random.normal(0, 1.5, len(dates))
                df_try = pd.DataFrame({"Date": dates, "IIP_Index": np.round(vals, 1)})

            elif sec == "gdp":
                quarters = [f"Q{i} 2024" for i in range(1, 5)] + [f"Q{i} 2025" for i in range(1, 5)]
                vals = [7.2, 7.5, 7.8, 8.0, 7.9, 7.7, 7.6, 7.5]
                df_try = pd.DataFrame({"Quarter": quarters[:len(vals)], "Real_GDP_Growth": vals})

            elif sec == "unemp":
                vals = np.linspace(7.5, 5.5, len(dates)) + np.random.normal(0, 0.2, len(dates))
                df_try = pd.DataFrame({"Date": dates, "Unemployment_Rate": np.round(vals, 2)})

# ========== LEFT COLUMN: CHARTS ==========
        with left:
            st.markdown(f"### {sec.upper()} — Visualisations")

            # Only proceed if we have a proper table
            if isinstance(df_try, pd.DataFrame) and not df_try.empty:
                date_col, value_col = detect_date_value_columns(df_try)

                if date_col and value_col:
                    tmp = df_try.copy()

                    # ---- Clean date column ----
                    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")

                    # ---- Clean numeric values (remove %, commas, text) ----
                    tmp[value_col] = (
                        tmp[value_col]
                        .astype(str)
                        .str.replace(r"[^\d\.\-]", "", regex=True)
                    )
                    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")

                    tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)

                    if tmp.empty:
                        st.info("Could not find clean numeric data to plot.")
                    else:
                        # ------- Latest metric card -------
                        last_row = tmp.iloc[-1]
                        latest_val = last_row[value_col]
                        latest_dt = last_row[date_col]

                        dt_str = (
                            latest_dt.strftime("%b %Y")
                            if not pd.isna(latest_dt)
                            else "Latest"
                        )

                        unit = "%" if ("%" in value_col.lower() or "rate" in value_col.lower()) else ""
                        metric_label = f"{sec.upper()} — latest"
                        metric_value = (
                            f"{latest_val:,.2f}{unit}"
                            if pd.notna(latest_val) else "N/A"
                        )
                        st.metric(metric_label, metric_value, dt_str)

                        # ===================================================
                        #  A N I M A T E D   L I N E   C H A R T
                        # ===================================================
                        st.markdown("#### Animated trend over time")

                        x_vals = tmp[date_col].dt.strftime("%b %Y").tolist()
                        y_vals = tmp[value_col].tolist()

                        fig_line = go.Figure(
                            data=[go.Scatter(x=[], y=[], mode="lines+markers")]
                        )

                        frames = []
                        for k in range(1, len(x_vals) + 1):
                            frames.append(
                                go.Frame(
                                    data=[
                                        go.Scatter(
                                            x=x_vals[:k],
                                            y=y_vals[:k],
                                            mode="lines+markers",
                                        )
                                    ],
                                    name=f"frame{k}",
                                )
                            )

                        fig_line.frames = frames
                        fig_line.update_layout(
                            xaxis_title="Period",
                            yaxis_title=value_col,
                            height=360,
                            template="plotly_white",
                            updatemenus=[
                                {
                                    "type": "buttons",
                                    "showactive": False,
                                    "x": 0.02,
                                    "y": 1.15,
                                    "buttons": [
                                        {
                                            "label": "Play",
                                            "method": "animate",
                                            "args": [
                                                None,
                                                {
                                                    "frame": {
                                                        "duration": 400,
                                                        "redraw": True,
                                                    },
                                                    "fromcurrent": True,
                                                },
                                            ],
                                        },
                                        {
                                            "label": "Pause",
                                            "method": "animate",
                                            "args": [
                                                [None],
                                                {
                                                    "frame": {"duration": 0},
                                                    "mode": "immediate",
                                                },
                                            ],
                                        },
                                    ],
                                }
                            ],
                        )
                        st.plotly_chart(fig_line, use_container_width=True)

                        # ------- Static bar chart (last 12 periods) -------
                        st.markdown("#### Last 12 periods (bar)")
                        recent = tmp.tail(12)
                        fig_bar = px.bar(
                            recent,
                            x=date_col,
                            y=value_col,
                            text_auto=".2f",
                        )
                        fig_bar.update_layout(
                            xaxis_title="Period (recent)",
                            yaxis_title=value_col,
                            height=320,
                            template="plotly_white",
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # ------- Smooth area chart (last 24 periods) -------
                        st.markdown("#### Last 24 periods (smooth area)")
                        area_df = tmp.tail(24)
                        if not area_df.empty:
                            fig_area = px.area(
                                area_df,
                                x=date_col,
                                y=value_col,
                            )
                            fig_area.update_layout(
                                xaxis_title="Period (recent)",
                                yaxis_title=value_col,
                                height=320,
                                template="plotly_white",
                            )
                            st.plotly_chart(fig_area, use_container_width=True)

                        # ------- Year-on-Year change chart (if enough history) -------
                        if len(tmp) >= 13:
                            st.markdown("#### Approx. Year-on-Year change (%)")

                            yoy_df = tmp.copy()
                            yoy_df["YoY_change_%"] = yoy_df[value_col].pct_change(periods=12) * 100
                            yoy_df = yoy_df.dropna(subset=["YoY_change_%"])

                            if not yoy_df.empty:
                                fig_yoy = px.line(
                                    yoy_df,
                                    x=date_col,
                                    y="YoY_change_%"
                                )
                                fig_yoy.update_layout(
                                    xaxis_title="Period",
                                    yaxis_title="YoY change (%)",
                                    height=320,
                                    template="plotly_white",
                                )
                                st.plotly_chart(fig_yoy, use_container_width=True)

                        # ------- Distribution of values (histogram) -------
                        st.markdown("#### Distribution of values")
                        hist_df = tmp[[value_col]].dropna()
                        if not hist_df.empty:
                            fig_hist = px.histogram(
                                hist_df,
                                x=value_col,
                                nbins=20,
                            )
                            fig_hist.update_layout(
                                xaxis_title=value_col,
                                yaxis_title="Frequency",
                                height=300,
                                template="plotly_white",
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)

                else:
                    st.info(
                        "Could not auto-detect date and value columns. "
                        "Showing raw data instead."
                    )
                    st.dataframe(df_try.head(20), use_container_width=True)
            else:
                st.info(
                    "No structured data available for this indicator. "
                    "Upload a CSV file in the admin panel."
                )
                
            # ========== RIGHT COLUMN: PRESS RELEASES + NEWS ==========
            with right:
                st.markdown(f"### {sec.upper()} — Press releases & News")
                if sec == "cpi":
                    show_press_and_news(
                        "CPI India",
                        resource_id=CPI_RESOURCE_ID,
                        uploaded_df=cpi_df_up,
                    )
                elif sec == "iip":
                    show_press_and_news(
                        "Index of Industrial Production India",
                        resource_id=IIP_RESOURCE_ID,
                        uploaded_df=iip_df_up,
                    )
                elif sec == "gdp":
                    show_press_and_news(
                        "GDP India",
                        resource_id=GDP_RESOURCE_ID,
                        uploaded_df=gdp_df_up,
                    )
                else:
                    show_press_and_news(
                        "Unemployment India",
                        resource_id=None,
uploaded_df=unemp_df_up,
)

render_macro_detail()

# ---------- Single-stock deep dive ----------
st.markdown("---")
st.markdown("## 💹 Stock — Single Symbol Deep Dive (Chart + Corporate Actions + Related News)")
st.markdown("Enter symbol in sidebar (e.g., RELIANCE.NS, AAPL, TCS.NS).")

if stock_input:
    # --- AUTO-DETECT CURRENCY FROM YFINANCE ---
    t = yf.Ticker(stock_input)
    info = t.info if hasattr(t, "info") else {}
    currency = info.get("currency", "USD")

    symbol_map = {
        "INR": "₹",
        "USD": "$",
        "EUR": "€",
        "GBP": "£"
    }
    ccy = symbol_map.get(currency, "")  # symbol to show with price

    st.markdown("### 📅 Select Time Range")

    tab_labels = ["1D", "3M", "6M", "1Y", "2Y", "3Y", "5Y"]
    tabs = st.tabs(tab_labels)
    period_map = {
        "1D": ("1d", "5m"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1d"),
        "2Y": ("2y", "1wk"),
        "3Y": ("3y", "1wk"),
        "5Y": ("5y", "1wk"),
    }

    if "selected_period" not in st.session_state:
        st.session_state["selected_period"] = "1Y"

    for label, tab in zip(tab_labels, tabs):
        with tab:
            if st.button(f"Select {label}", key=f"tab_{label}"):
                st.session_state["selected_period"] = label
                st.experimental_rerun()

    selected_label = st.session_state["selected_period"]
    period, interval = period_map[selected_label]

    # --- Fetch stock data ---
    with st.spinner(f"Fetching {stock_input} data for {selected_label}..."):
        data = yf.download(
            stock_input,
            period=period,
            interval=interval,
            progress=False,
        )

    if data.empty:
        st.warning("⚠️ No stock data found. Try another symbol or add .NS for Indian stocks.")
    else:
        data = data.reset_index()
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest

        current_price = float(latest["Close"])
        prev_price = float(prev["Close"])
        change_val = current_price - prev_price
        change_pct = (change_val / prev_price) * 100 if prev_price else 0.0
        open_price = float(latest["Open"])
        high_price = float(latest["High"])
        low_price = float(latest["Low"])
        volume = int(latest["Volume"])

        color = "green" if change_val > 0 else "red" if change_val < 0 else "gray"
        sentiment = (
            "Bullish 📈" if change_val > 0 else
            "Bearish 📉" if change_val < 0 else
            "Neutral ⚖️"
        )

        # --- Current snapshot (with animated main price) ---
        st.markdown(f"### {stock_input} — Current Snapshot")
        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            animate_metric(
                label="Price",
                value=current_price,
                delta=f"{change_val:+.2f}",
                state_key=f"stock_price_{stock_input}",
            )

        c2.metric("Change (%)", f"{change_pct:+.2f}%")
        c3.metric("Open",  f"{ccy}{open_price:,.2f}")
        c4.metric("High",  f"{ccy}{high_price:,.2f}")
        c5.metric("Low",   f"{ccy}{low_price:,.2f}")
        c6.metric("Volume", f"{volume:,}")
        st.caption(f"🕒 Last Updated: {latest['Date']} | Sentiment: {sentiment}")

        # --- Price trend chart ---
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Close"],
                mode="lines",
                name="Price",
                line=dict(color=color, width=2),
            )
        )
        fig.update_layout(
            title=f"{stock_input} — {selected_label} Trend",
            yaxis_title=f"Price ({ccy})",
            xaxis_title="Date",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Moving averages chart ---
        st.markdown("### 📊 Moving Averages (Trend Analysis)")
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["MA200"] = data["Close"].rolling(window=200).mean()

        show_ma20 = st.checkbox("Show MA20 (Short-term)", value=True)
        show_ma50 = st.checkbox("Show MA50 (Medium-term)", value=True)
        show_ma200 = st.checkbox("Show MA200 (Long-term)", value=False)

        fig_ma = go.Figure()
        fig_ma.add_trace(
            go.Scatter(
                x=data["Date"],
                y=data["Close"],
                mode="lines",
                line=dict(color=color, width=2),
                name="Price",
            )
        )
        if show_ma20:
            fig_ma.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["MA20"],
                    mode="lines",
                    line=dict(width=1.8, dash="dot"),
                    name="MA20",
                )
            )
        if show_ma50:
            fig_ma.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["MA50"],
                    mode="lines",
                    line=dict(width=1.8, dash="dot"),
                    name="MA50",
                )
            )
        if show_ma200:
            fig_ma.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["MA200"],
                    mode="lines",
                    line=dict(width=1.8, dash="dot"),
                    name="MA200",
                )
            )

        fig_ma.update_layout(
            title=f"{stock_input} — Moving Averages",
            yaxis_title=f"Price ({ccy})",
            xaxis_title="Date",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_ma, use_container_width=True)

        # --- Corporate actions + events in ONE table ---
        st.markdown("### 🏢 Corporate Actions & Events (Summary)")

        sa = fetch_stock_actions(stock_input)
        divs = sa.get("dividends")
        splits = sa.get("splits")
        events = sa.get("events", [])
        news_list = sa.get("news", [])

        rows = []

        # 1) Dividends
        if not getattr(divs, "empty", True):
        
            for dt, val in divs.tail(10).items():
                rows.append(
                    {
                        "Category": "DIVIDEND",
                        "Date": pd.to_datetime(dt),
                        "Title / Detail": f"Dividend {val:.2f} per share",
                        "Extra": "",
                    }
                )

        # 2) Splits
        if not getattr(splits, "empty", True):
            for dt, ratio in splits.tail(10).items():
                rows.append(
                    {
                        "Category": "SPLIT",
                        "Date": pd.to_datetime(dt),
                        "Title / Detail": f"Split ratio {ratio}",
                        "Extra": "",
                    }
                )

        def classify_event(ev_type: str, detail: str) -> str:
            t = (ev_type or "").upper()
            d = (detail or "").lower()

            if t in {
                "BONUS",
                "BONUS ISSUE",
                "RIGHTS",
                "RIGHTS ISSUE",
                "BUYBACK",
                "OFS",
                "BOND ISSUE",
            }:
                return t

            if "bonus" in d:
                return "BONUS ISSUE"
            if "rights issue" in d:
                return "RIGHTS ISSUE"
            if "buyback" in d or "buy-back" in d:
                return "BUYBACK"
            if "offer for sale" in d or "ofs" in d:
                return "OFS"
            if "bond" in d or "debenture" in d:
                return "BOND ISSUE"
            if "dividend" in d:
                return "DIVIDEND"
            if "split" in d or "sub-division" in d:
                return "SPLIT"

            return t or "EVENT"

        # 3) Structured events
        for ev in events[-20:]:
            ev_type = ev.get("type", "")
            detail = ev.get("detail", "")
            cat = classify_event(ev_type, detail)

            rows.append(
                {
                    "Category": cat,
                    "Date": ev.get("date"),
                    "Title / Detail": detail,
                    "Extra": ev.get("source", ""),
                }
            )

        # 4) News as event entries
        for n in news_list[:20]:
            headline = n.get("title", "")
            pub = n.get("publisher", "")
            ts = n.get("providerPublishTime")
            cat = classify_event("EVENT NEWS", headline)

            rows.append(
                {
                    "Category": cat,
                    "Date": pd.to_datetime(ts, unit="s", errors="ignore")
                    if ts
                    else "",
                    "Title / Detail": headline,
                    "Extra": pub,
                }
            )

# ---------- 5) Show table ----------
if rows:
    df_actions = pd.DataFrame(rows)

    # 👉 Convert Date column safely
    if "Date" in df_actions.columns:
        df_actions["Date"] = pd.to_datetime(df_actions["Date"], errors="coerce")
        df_actions = (
            df_actions
            .sort_values(by=["Date"], ascending=False, na_position="last")
            .reset_index(drop=True)
        )
    else:
        df_actions = df_actions.reset_index(drop=True)

    st.dataframe(df_actions, use_container_width=True)
else:
    st.info("No corporate actions / events found for this symbol.")
    
# --- Corporate event news with sentiment ---
st.markdown("### 📰 Corporate Event News (Sentiment)")

# Build a search query for this company
company_query = (
    info.get("shortName")
    or info.get("longName")
    or stock_input
)

# Use our global fetch_news() function
evt_news = fetch_news(company_query, n=10)

if evt_news:
    for a in evt_news:
        title = (a.get("title") or "News item").strip()
        link = a.get("url") or ""
        publisher = a.get("source") or ""
        summary = a.get("summary") or ""

        # Sentiment on title + summary
        label, score = sentiment_label(f"{title} {summary}")
        color = (
            PALETTE["pos"] if label == "positive"
            else PALETTE["neg"] if label == "negative"
            else PALETTE["neu"]
        )

        st.markdown(
            f"- [{title}]({link})  \n"
            f"  <span style='color:{color}; font-weight:600'>{label.upper()}</span> ({score:+.2f}) · {publisher}",
            unsafe_allow_html=True,
        )
else:
    st.info("No recent company news found for this symbol.")
    
# ---------- Footer & debug ----------
st.markdown("---")
st.markdown(f"<div style='color:{PALETTE['teal']}'>Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>", unsafe_allow_html=True)

with st.expander("Show internal debug log"):
    for r in st.session_state["_log"][-200:]:
        st.text(r)

st.markdown("---")
st.markdown("**Tips:** set NEWSAPI_KEY and DATA_GOV_API_KEY in .env or Streamlit Secrets for better data. Upload CSVs if automatic fetch fails.")
