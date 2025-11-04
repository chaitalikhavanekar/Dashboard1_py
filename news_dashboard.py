# news_dashboard.py
"""
India Economic Intelligence — Full app
Contains:
- News & Insights (NewsAPI or Google RSS fallback)
- Personalized trending feed + interest learning
- Market indices snapshot
- Single-stock view + corporate actions
- MOSPI / data.gov integration for CPI/IIP/GDP micro-charts (auto if resource IDs provided) + CSV upload fallback
- Auto-news newsletter generator (3-4 bullets) + download + optional email send via SMTP
- Auto-refresh with visible countdown (uses streamlit-autorefresh if installed)
- Color palette: SkyBlue (#C8D9E6), Beige (#F5EFEB), Navy (#2F4156), Teal (#567C8D), White
Notes:
- Create a `.env` to store keys (optional). Example keys:
  NEWSAPI_KEY=your_key
  DATA_GOV_API_KEY=your_data_gov_key
  CPI_RESOURCE_ID=...
  IIP_RESOURCE_ID=...
  GDP_RESOURCE_ID=...
  SMTP_HOST=...
  SMTP_PORT=...
  SMTP_USER=...
  SMTP_PASS=...
"""

# Standard libs
import os
import time
import textwrap
from datetime import datetime, timezone
from collections import Counter, defaultdict
from io import BytesIO

# Data libs
import requests
import feedparser
import requests_cache
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# NLP
from textblob import TextBlob

# Streamlit UI
import streamlit as st

# optional autorefresh import
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTOREF = True
except Exception:
    HAS_AUTOREF = False

# dotenv (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# cache HTTP requests to reduce repeated network calls
requests_cache.install_cache("news_cache", expire_after=180)

# ---------------------------
# CONFIG / PALETTE / KEYS
# ---------------------------
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

# API / resource IDs (use .env or Streamlit Secrets)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "").strip()
CPI_RESOURCE_ID = os.getenv("CPI_RESOURCE_ID", "").strip()
IIP_RESOURCE_ID = os.getenv("IIP_RESOURCE_ID", "").strip()
GDP_RESOURCE_ID = os.getenv("GDP_RESOURCE_ID", "").strip()

# SMTP optional for sending newsletter
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "0")) if os.getenv("SMTP_PORT") else None
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()

# Indices mapping
INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "S&P 500": "^GSPC",
}

# caching TTLs
NEWS_TTL = 120
MARKET_TTL = 60
MACRO_TTL = 3600

# helper logging to session state for debugging
if "_log" not in st.session_state:
    st.session_state["_log"] = []

def log(msg):
    st.session_state["_log"].append(f"{datetime.utcnow().isoformat()}  {msg}")

# ---------------------------
# CSS and page config
# ---------------------------
st.set_page_config(page_title="India Economic Intelligence", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
:root {{
  --sky: {PALETTE['sky']};
  --beige: {PALETTE['beige']};
  --navy: {PALETTE['navy']};
  --teal: {PALETTE['teal']};
  --card: {PALETTE['card']};
  --pos: {PALETTE['pos']};
  --neg: {PALETTE['neg']};
  --neu: {PALETTE['neu']};
}}
/* app background */
[data-testid="stAppViewContainer"] {{
  background: linear-gradient(180deg, var(--sky) 0%, var(--beige) 100%);
}}
/* headings */
h1, h2, h3, h4 {{
  color: var(--navy);
}}
/* card */
.card {{
  background: var(--card);
  border-radius: 12px;
  padding: 12px;
  margin-bottom: 10px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}}
.small-muted {{ color: var(--teal); font-size:0.95em; }}
.sent-badge {{
  display:inline-block; padding:4px 8px; border-radius:10px; color:white; font-weight:600; font-size:12px;
}}
/* sidebar slight tint */
[data-testid="stSidebar"] {{
  background-color: rgba(255,255,255,0.6);
}}
/* reduce big default spacing in some cases */
.css-1d391kg .block-container {{ padding-top: 8px; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utility functions
# ---------------------------

def safe_json_get(url, params=None, headers=None, timeout=12):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"HTTP error: {e} | url={url}")
        return None

# News fetching: NewsAPI first, fallback to Google News RSS
@st.cache_data(ttl=NEWS_TTL)
def fetch_news_newsapi(query, n=10):
    if not NEWSAPI_KEY:
        return None
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "language": "en", "pageSize": n, "sortBy": "publishedAt", "apiKey": NEWSAPI_KEY}
    j = safe_json_get(url, params=params)
    if not j or j.get("status") != "ok":
        return None
    articles = []
    for art in j.get("articles", [])[:n]:
        articles.append({
            "title": art.get("title"),
            "summary": art.get("description") or art.get("content") or "",
            "url": art.get("url"),
            "source": (art.get("source") or {}).get("name"),
            "publishedAt": art.get("publishedAt")
        })
    return articles

@st.cache_data(ttl=NEWS_TTL)
def fetch_news_google_rss(query, n=10, country="IN"):
    q = requests.utils.requote_uri(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-{country}&gl={country}&ceid={country}:en"
    feed = feedparser.parse(url)
    items = []
    for ent in feed.entries[:n]:
        items.append({
            "title": ent.get("title"),
            "summary": ent.get("summary", ""),
            "url": ent.get("link"),
            "source": (ent.get("source") or {}).get("title") if ent.get("source") else None,
            "publishedAt": ent.get("published") or ent.get("published_parsed")
        })
    return items

def fetch_news(query, n=8):
    # try NewsAPI first
    res = fetch_news_newsapi(query, n=n) if NEWSAPI_KEY else None
    if res:
        return res
    # fallback
    return fetch_news_google_rss(query, n=n)

def sentiment_score(text):
    try:
        tb = TextBlob(text or "")
        score = round(tb.sentiment.polarity, 3)
        if score >= 0.05:
            label = "positive"
        elif score <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return {"label": label, "score": score}
    except Exception as e:
        log(f"sentiment error: {e}")
        return {"label": "neutral", "score": 0.0}

# Market snapshot
@st.cache_data(ttl=MARKET_TTL)
def fetch_index_snapshot():
    out = {}
    for name, sym in INDICES.items():
        try:
            df = yf.download(sym, period="3d", interval="1d", progress=False, threads=False)
            if df is None or df.empty:
                out[name] = {"last": None, "pct": None}
            else:
                last = float(df["Close"].iloc[-1])
                prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
                pct = (last - prev) / prev * 100 if prev != 0 else 0.0
                out[name] = {"last": last, "pct": pct}
        except Exception as e:
            out[name] = {"last": None, "pct": None}
            log(f"index fetch error {name}: {e}")
    return out

# yfinance helpers for stocks
@st.cache_data(ttl=MARKET_TTL)
def fetch_stock_history(sym, period="1y", interval="1d"):
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={"Close":"close"}, inplace=True)
        return df
    except Exception as e:
        log(f"stock history error {sym}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=MARKET_TTL)
def fetch_stock_actions(sym):
    try:
        t = yf.Ticker(sym)
        divs = t.dividends if hasattr(t, "dividends") else pd.Series(dtype=float)
        splits = t.splits if hasattr(t, "splits") else pd.Series(dtype=float)
        # yfinance .news is not always available; safe-get if present
        news = []
        try:
            raw_news = t.news
            if isinstance(raw_news, list):
                for item in raw_news[:8]:
                    news.append({"title": item.get("title"), "link": item.get("link")})
        except Exception:
            pass
        return {"dividends": divs, "splits": splits, "news": news}
    except Exception as e:
        log(f"stock actions error {sym}: {e}")
        return {"dividends": pd.Series(dtype=float), "splits": pd.Series(dtype=float), "news": []}

# Data.gov fetch for MOSPI resources
@st.cache_data(ttl=MACRO_TTL)
def fetch_data_gov_resource(resource_id, limit=1000, api_key=None):
    if not resource_id:
        return None
    key = api_key or DATA_GOV_API_KEY
    if not key:
        return None
    try:
        base = f"https://api.data.gov.in/resource/{resource_id}.json"
        params = {"api-key": key, "limit": limit}
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log(f"data.gov fetch error {resource_id}: {e}")
        return None

# Simple trending extraction
def extract_trending_terms(headlines, top_n=8):
    stop = set(["the","and","for","with","from","that","this","are","was","will","have","has","india","govt","government"])
    words = []
    for h in headlines:
        if not h: continue
        for token in h.lower().split():
            token = "".join(ch for ch in token if ch.isalpha())
            if len(token) > 3 and token not in stop:
                words.append(token)
    return [w for w,_ in Counter(words).most_common(top_n)]

# Personalization logic (simple)
def init_personalization():
    if "prefs" not in st.session_state:
        st.session_state["prefs"] = ["rbi","infrastructure","inflation"]
    if "click_counts" not in st.session_state:
        st.session_state["click_counts"] = defaultdict(int)

def record_click(article_id):
    st.session_state["click_counts"][article_id] += 1

def score_article_for_user(article, interests, trending):
    text = (article.get("title","") + " " + (article.get("summary") or "")).lower()
    score = 0
    for it in interests:
        if it.lower() in text:
            score += 2
    for t in trending:
        if t in text:
            score += 1
    # clicks boost if same title tuple present
    aid = article.get("url") or article.get("title")
    score += st.session_state["click_counts"].get(aid, 0) * 0.5
    return score

# Newsletter generator
def generate_newsletter(headlines, market_snapshot, macro_bullets=[]):
    bullets = []
    # 1 macro bullet if available
    bullets.extend(macro_bullets[:1])
    # pick top 3 headlines based on sentiment + recency
    if headlines:
        for h in headlines[:4]:
            title = h.get("title","")
            short = textwrap.shorten(title, width=130, placeholder="...")
            label = sentiment_score = sentiment_score = sentiment_score = sentiment_score  # placeholder
            bullets.append(short)
    # ensure 3-4 bullets
    return bullets[:4]

# small safe formatter for dates
def format_dt(dt):
    if not dt:
        return ""
    if isinstance(dt, str):
        return dt
    try:
        return pd.to_datetime(dt).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(dt)

# ---------------------------
# UI: Sidebar controls
# ---------------------------
init_personalization()

st.sidebar.title("Controls & Settings")
st.sidebar.markdown("**Search & Refresh**")
search_query = st.sidebar.text_input("Search query (news)", value="India economy OR RBI OR inflation OR GDP OR infrastructure")
headlines_count = st.sidebar.slider("Headlines to show", min_value=3, max_value=20, value=6)
auto_refresh_label = st.sidebar.selectbox("Auto-refresh", options=["Off","30s","1m","5m"], index=2)
stock_symbol = st.sidebar.text_input("Single stock (one symbol):", value="RELIANCE.NS")
st.sidebar.markdown("---")

# personalization controls
st.sidebar.markdown("**Your interests (used to personalize feed)**")
prefs = st.sidebar.multiselect("Select interests (you can add)", options=["RBI","inflation","infrastructure","startups","banks","policy","GDP","employment","stock"], default=st.session_state["prefs"])
if st.sidebar.button("Save interests"):
    st.session_state["prefs"] = prefs

st.sidebar.markdown("---")
st.sidebar.markdown("🔧 *Advanced (optional)*")
st.sidebar.text("Set env vars in .env or Streamlit Secrets for:\nNEWSAPI_KEY, DATA_GOV_API_KEY, CPI_RESOURCE_ID, IIP_RESOURCE_ID, GDP_RESOURCE_ID")

# auto-refresh handling
interval_map = {"Off":0, "30s":30, "1m":60, "5m":300}
interval_seconds = interval_map.get(auto_refresh_label, 0)
if HAS_AUTOREF and interval_seconds>0:
    refresh_count = st_autorefresh(interval=interval_seconds*1000, key="autorefresh")
    st.sidebar.markdown(f"Auto-refresh ticks: {refresh_count}")
else:
    if interval_seconds>0:
        st.sidebar.info(f"Auto-refresh set to {auto_refresh_label} — install 'streamlit-autorefresh' for page auto-reload.")

# manual refresh button
if st.sidebar.button("Refresh now"):
    requests_cache.clear()
    st.experimental_rerun()

# ---------------------------
# Top header & indices snapshot
# ---------------------------
st.markdown(f"<h1>📰 News & Insights — India Economic Intelligence</h1>", unsafe_allow_html=True)
st.markdown(f"<div class='small-muted'>Live economic, policy, infrastructure, employment & market news — personalized feed</div>", unsafe_allow_html=True)
st.markdown("---")

# Fetch indices snapshot (cached)
with st.spinner("Fetching market snapshot..."):
    indices = fetch_index_snapshot()

# Show indices as small tiles
cols = st.columns(len(INDICES))
for i,(name, sym) in enumerate(INDICES.items()):
    col = cols[i]
    val = indices.get(name, {})
    last = val.get("last")
    pct = val.get("pct")
    if last is None:
        col.markdown(f"<div class='card'><div style='font-weight:700'>{name}</div><div style='color:#777'>N/A</div></div>", unsafe_allow_html=True)
    else:
        color = PALETTE["pos"] if pct>=0 else PALETTE["neg"]
        arrow = "▲" if pct>=0 else "▼"
        col.markdown(f"<div class='card'><div style='font-weight:700'>{name}</div><div style='font-size:18px'>{last:,.2f}</div><div style='color:{color}; font-weight:700'>{arrow} {pct:+.2f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------
# Fetch news for query & build sentiment + personalization
# ---------------------------
with st.spinner("Fetching news..."):
    news_items = fetch_news(search_query, n=headlines_count)
    # attach sentiment
    for it in news_items:
        s = sentiment_score = sentiment_score = None  # placeholder, we will compute below
    # do sentiment separately
    enriched = []
    for it in news_items:
        txt = (it.get("title","") or "") + ". " + (it.get("summary") or "")
        s = sentiment_score(txt)
        it["sent_label"] = s["label"]
        it["sent_score"] = s["score"]
        enriched.append(it)
    news_items = enriched

# trending + personalization scoring
headlines_text = [it.get("title","") for it in news_items]
trending_terms = extract_trending_terms(headlines_text, top_n=8)
# score for personalization
for it in news_items:
    it["_user_score"] = score_article = score_article_for_user = None  # placeholder

# assign personalized score properly
for it in news_items:
    it["_user_score"] = score_article_for_user(it, prefs if prefs else st.session_state["prefs"], trending_terms)

# sort a combined feed: first by user score desc then by recency (if available)
def parse_pubdate(x):
    p = x.get("publishedAt") or x.get("published")
    try:
        return pd.to_datetime(p)
    except Exception:
        return pd.Timestamp.min

news_items_sorted = sorted(news_items, key=lambda x: (x["_user_score"], parse_pubdate(x)), reverse=True)

# ---------------------------
# Layout: two columns - main news + right pane
# ---------------------------
main_col, side_col = st.columns([3, 1])

with main_col:
    st.markdown("<div style='font-weight:700; font-size:20px'>Top headlines</div>", unsafe_allow_html=True)
    if not news_items_sorted:
        st.info("No news found. Try another query or set NEWSAPI_KEY for richer results.")
    # list headlines compactly with sentiment badge inline
    for idx, art in enumerate(news_items_sorted, start=1):
        title = art.get("title") or ""
        summary = art.get("summary") or ""
        url = art.get("url") or "#"
        src = art.get("source") or ""
        pub = art.get("publishedAt") or art.get("published") or ""
        sent = art.get("sent_label")
        score = art.get("sent_score")
        badge_color = PALETTE["pos"] if sent=="positive" else (PALETTE["neg"] if sent=="negative" else PALETTE["neu"])
        badge_html = f"<span class='sent-badge' style='background:{badge_color}'>{sent.upper()}</span>"
        # clickable card
        article_id = url or title
        # display compact
        st.markdown(f"""
            <div class='card'>
              <div style='display:flex; align-items:center; justify-content:space-between;'>
                <div style='flex:1'>
                  <a href="{url}" target="_blank" style="color:{PALETTE['navy']}; font-weight:700; font-size:15px; text-decoration:none;">{idx}. {title}</a>
                  <div class='small-muted'>{src} · {format_dt(pub)}</div>
                </div>
                <div style='margin-left:8px; text-align:right;'>
                  {badge_html}
                  <div style='font-size:12px; color:{PALETTE['teal']}; margin-top:6px;'>Score: {score:+.2f}</div>
                </div>
              </div>
              <div style='margin-top:8px; color:#333'>{summary}</div>
              <div style='margin-top:8px'><a href="{url}" target="_blank">Read full article →</a></div>
            </div>
        """, unsafe_allow_html=True)
        # small click tracker (if user clicks the "Read full article" they open new tab; we also provide a 'Mark read' button)
        c1, c2 = st.columns([4,1])
        with c2:
            if st.button("Mark read", key=f"mr_{idx}"):
                record_click(article_id)
                st.experimental_rerun()

with side_col:
    st.markdown("<div style='font-weight:700; font-size:18px'>Trending</div>", unsafe_allow_html=True)
    if trending_terms:
        for t in trending_terms:
            st.markdown(f"- {t}")
    else:
        st.write("—")
    st.markdown("---")
    # personalized picks
    st.markdown("<div style='font-weight:700'>For you (top picks)</div>", unsafe_allow_html=True)
    top_for_you = sorted(news_items_sorted, key=lambda x: x["_user_score"], reverse=True)[:4]
    if not top_for_you:
        st.write("No personalized picks available.")
    else:
        for a in top_for_you:
            st.markdown(f"- <a href='{a.get('url')}' target='_blank'>{a.get('title')}</a>", unsafe_allow_html=True)
    st.markdown("---")
    # small quick filters
    st.markdown("### Quick filters")
    if st.button("Show only positive"):
        st.session_state["_filter"] = "positive"
        st.experimental_rerun()
    if st.button("Show only negative"):
        st.session_state["_filter"] = "negative"
        st.experimental_rerun()
    if st.button("Reset filter"):
        st.session_state.pop("_filter", None)
        st.experimental_rerun()

# apply filter if set
flt = st.session_state.get("_filter")
if flt:
    news_items_sorted = [n for n in news_items_sorted if n.get("sent_label")==flt]

# ---------------------------
# Newsletter generator (auto summary)
# ---------------------------
st.markdown("---")
st.markdown("### 📝 Auto Newsletter — 3–4 bullet brief")
# newsletter building: pick top personalized plus 1 macro if available
macro_bullets = []
# try to get a quick CPI bullet if CPI data available
cpi_brief = None
if CPI_RESOURCE_ID and DATA_GOV_API_KEY:
    j = fetch_data_gov_resource(CPI_RESOURCE_ID, limit=5)
    if j and j.get("records"):
        recs = j.get("records")
        # try to find a column with cpi value and date
        df_c = pd.DataFrame(recs)
        date_col = next((c for c in df_c.columns if "date" in c.lower() or "month" in c.lower()), None)
        val_col = next((c for c in df_c.columns if any(x in c.lower() for x in ["value","cpi","index"])), None)
        if date_col and val_col:
            latest = df_c.sort_values(date_col).iloc[-1]
            cpi_brief = f"CPI: {val_col} = {latest[val_col]} ({latest[date_col]})"
            macro_bullets.append(cpi_brief)

# build headlines brief from top_for_you
nl_lines = []
if top_for_you:
    for a in top_for_you[:3]:
        text = a.get("title") or ""
        nl_lines.append(textwrap.shorten(text, width=140, placeholder="..."))

# combine
newsletter_lines = []
if macro_bullets:
    newsletter_lines.extend(macro_bullets[:1])
newsletter_lines.extend(nl_lines)
if not newsletter_lines:
    newsletter_lines = ["No news available to summarize."]

newsletter_text = "Daily Economic Brief — Auto-generated\n\n" + "\n".join(f"{i+1}. {l}" for i,l in enumerate(newsletter_lines))

# show newsletter and download/send controls
st.text_area("Newsletter (editable)", value=newsletter_text, height=200, key="newsletter_area")
st.download_button("Download newsletter (TXT)", data=newsletter_text.encode("utf-8"), file_name="economic_brief.txt")

# optional SMTP send
if SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS:
    st.markdown("Send newsletter via email (SMTP configured in env)")
    to_addr = st.text_input("Send to (comma-separated emails)", value="")
    if st.button("Send newsletter"):
        import smtplib, ssl
        from email.message import EmailMessage
        msg = EmailMessage()
        msg["Subject"] = "Daily Economic Brief"
        msg["From"] = SMTP_USER
        msg["To"] = [a.strip() for a in to_addr.split(",") if a.strip()]
        msg.set_content(st.session_state["newsletter_area"])
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as smtp:
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
            st.success("Newsletter sent.")
        except Exception as e:
            st.error(f"Send failed: {e}")
else:
    st.info("SMTP not configured. To enable email sending, set SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS in .env or Secrets.")

# ---------------------------
# MOSPI / MACRO micro charts
# ---------------------------
st.markdown("---")
st.markdown("## 📈 Macro indicators (CPI / IIP / GDP) — auto (data.gov) or upload fallback")
st.markdown("If automatic fetch fails, please upload CSV/XLSX with 'date' and 'value' columns for plotting.")

def load_macro(kind, resource_env_id):
    key = resource_env_id
    uploaded = st.file_uploader(f"Upload {kind.upper()} CSV/XLSX (fallback) - optional", type=["csv","xlsx"], key=f"up_{kind}")
    # try data.gov resource if available
    df = None; source = None
    if key and DATA_GOV_API_KEY:
        j = fetch_data_gov_resource(key, limit=2000, api_key=DATA_GOV_API_KEY)
        if j and j.get("records"):
            df = pd.DataFrame(j["records"])
            source = f"data.gov resource {key}"
    if df is None and uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            source = f"uploaded {uploaded.name}"
        except Exception as e:
            st.error(f"Upload parse error for {kind}: {e}")
            return None, None
    return df, source

cpi_df, cpi_src = load_macro("cpi", CPI_RESOURCE_ID)
iip_df, iip_src = load_macro("iip", IIP_RESOURCE_ID)
gdp_df, gdp_src = load_macro("gdp", GDP_RESOURCE_ID)

def auto_plot_macro(df, kind_label):
    if df is None:
        st.info(f"No {kind_label} dataset available to plot.")
        return
    cols = df.columns.tolist()
    date_col = next((c for c in cols if "date" in c.lower() or "month" in c.lower()), None)
    value_col = next((c for c in cols if any(x in c.lower() for x in ["value","index","cpi","iip","gdp","amount","level"])), None)
    if date_col and value_col:
        try:
            tmp = df[[date_col, value_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col, value_col])
            tmp = tmp.sort_values(date_col)
            fig = px.line(tmp, x=date_col, y=value_col, title=f"{kind_label} — {value_col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plotting error: {e}")
    else:
        st.dataframe(df.head(6))
        st.info("Could not auto-detect date/value columns. Upload CSV with 'date' and 'value' columns for auto-plotting.")

# show three macro cards side-by-side
mcols = st.columns(3)
with mcols[0]:
    st.markdown("### CPI")
    if cpi_src:
        st.caption(cpi_src)
    auto_plot_macro(cpi_df, "CPI")
with mcols[1]:
    st.markdown("### IIP")
    if iip_src:
        st.caption(iip_src)
    auto_plot_macro(iip_df, "IIP")
with mcols[2]:
    st.markdown("### GDP")
    if gdp_src:
        st.caption(gdp_src)
    auto_plot_macro(gdp_df, "GDP")

# ---------------------------
# Per-stock detailed view (bottom)
# ---------------------------
st.markdown("---")
st.markdown("## 💹 Stock — single symbol deep dive")
st.markdown("Enter symbol in sidebar (e.g., RELIANCE.NS for Indian stocks). Chart, corporate actions and related news will appear.")

if stock_symbol:
    with st.spinner(f"Fetching {stock_symbol} data..."):
        stock_history = fetch_stock_history(stock_symbol, period="1y")
        stock_actions = fetch_stock_actions(stock_symbol)
    if stock_history.empty:
        st.warning("No history available for this symbol. Check ticker format (e.g., add .NS for India).")
    else:
        latest = stock_history["close"].iloc[-1]
        prev = stock_history["close"].iloc[-2] if len(stock_history)>1 else latest
        pct = (latest - prev)/prev*100 if prev!=0 else 0.0
        st.metric(label=f"{stock_symbol} Latest", value=f"{latest:,.2f}", delta=f"{pct:+.2f}%")
        fig = px.line(stock_history, x="Date", y="close", title=f"{stock_symbol} price (1 year)")
        color = PALETTE["pos"] if pct>=0 else PALETTE["neg"]
        fig.update_traces(line=dict(color=color, width=2))
        st.plotly_chart(fig, use_container_width=True)
        # corporate actions
        st.markdown("### Corporate actions (dividends & splits)")
        divs = stock_actions.get("dividends")
        splits = stock_actions.get("splits")
        if divs is not None and not getattr(divs, "empty", True):
            try:
                ddf = divs.reset_index()
                ddf.columns = ["Date","Dividend"]
                ddf["Date"] = pd.to_datetime(ddf["Date"], errors="coerce")
                st.dataframe(ddf.sort_values("Date", ascending=False).head(8))
            except Exception:
                st.write(divs.tail(8))
        else:
            st.info("No dividend records found (via yfinance).")
        if splits is not None and not getattr(splits, "empty", True):
            try:
                sdf = splits.reset_index()
                sdf.columns = ["Date","Split"]
                sdf["Date"] = pd.to_datetime(sdf["Date"], errors="coerce")
                st.dataframe(sdf.sort_values("Date", ascending=False).head(8))
            except Exception:
                st.write(splits.tail(8))
        else:
            st.info("No split records found (via yfinance).")
        # related news
        st.markdown("### Related news (ticker search fallback)")
        related = fetch_news(search_query + " " + stock_symbol, n=6)
        if not related:
            st.info("No related news found.")
        else:
            for r in related:
                st.markdown(f"- <a href='{r.get('url')}' target='_blank'>{r.get('title')}</a>", unsafe_allow_html=True)

# ---------------------------
# Footer + logs + help
# ---------------------------
st.markdown("---")
st.markdown("#### Help & Troubleshooting")
st.markdown("""
- If News results are limited, add your `NEWSAPI_KEY` in `.env` or Streamlit Secrets.
- For MOSPI automatic data: set `DATA_GOV_API_KEY` and resource IDs `CPI_RESOURCE_ID`, `IIP_RESOURCE_ID`, `GDP_RESOURCE_ID`.
- For email sending: set SMTP variables in env (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS).
- If a package fails on deploy, check `requirements.txt` and deployment logs.
""")

with st.expander("Show debug log"):
    for r in st.session_state["_log"][-80:]:
        st.text(r)

# End of file
