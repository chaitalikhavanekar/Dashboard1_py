# news_dashboard.py
"""
News & Insights — Enhanced
Features added:
- MOSPI/API + Data.gov fallback for CPI, IIP, GDP micro-charts
- Newsletter auto-summary (3-4 bullets) from top headlines + indicators
- Auto-refresh using streamlit-autorefresh with visible countdown (optional)
- Per-stock search (one symbol): price, 1yr chart, dividends, splits, related news
- Color palette: Sky Blue / Beige / Navy / Teal; charts use green/red for gains/losses
Notes:
- Optional API keys in environment/Streamlit secrets: NEWSAPI_KEY, DATA_GOV_API_KEY, CPI_RESOURCE_ID, IIP_RESOURCE_ID, GDP_RESOURCE_ID, TRADINGECONOMICS_KEY
- Use Google News RSS fallback when NEWSAPI_KEY not provided
"""
import os
import time
import math
import requests
import feedparser
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from datetime import datetime, timezone
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import streamlit as st

# --- Optional autorefresh import (non-fatal) ---
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_ST_AUTOREFRESH = True
except Exception:
    HAS_ST_AUTOREFRESH = False

# -----------------------------
# ---------- CONFIG ----------
# -----------------------------
st.set_page_config(page_title="📰 News & Insights + MOSPI", layout="wide")

# Color palette (user)
PALETTE = {
    "bg1": "#C8D9E6",   # SkyBlue
    "bg2": "#F5EFE8",   # Beige
    "navy": "#2F4156",  # Navy heading
    "teal": "#567C8D",  # Teal subtext
    "white": "#FFFFFF",
    "pos": "#00C49F",   # green for positive
    "neg": "#FF4C4C",   # red
    "neu": "#F5B041"    # amber
}

# Indices mapping
INDICES = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "NASDAQ": "^IXIC",
    "DOW JONES": "^DJI",
    "S&P 500": "^GSPC"
}

# Cache TTL for network calls
CACHE_TTL = 180  # seconds

# API keys (optional)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "").strip()
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "").strip()
TRADINGECONOMICS_KEY = os.getenv("TRADINGECONOMICS_KEY", "").strip()

# MOSPI / data.gov resource ids (set via environment or Streamlit secrets)
CPI_RESOURCE_ID = os.getenv("CPI_RESOURCE_ID", "").strip()   # optional
IIP_RESOURCE_ID = os.getenv("IIP_RESOURCE_ID", "").strip()
GDP_RESOURCE_ID = os.getenv("GDP_RESOURCE_ID", "").strip()

# Sentiment analyzer
ANALYZER = SentimentIntensityAnalyzer()

# -----------------------------
# ---------- HELPERS ----------
# -----------------------------

def classify_sentiment(text: str):
    if not text:
        return ("neutral", 0.0)
    s = ANALYZER.polarity_scores(text)
    c = s["compound"]
    if c >= 0.05:
        return ("positive", c)
    elif c <= -0.05:
        return ("negative", c)
    else:
        return ("neutral", c)

@st.cache_data(ttl=CACHE_TTL)
def fetch_google_rss(query: str, country="IN", max_items=10):
    """Google News RSS fallback"""
    try:
        q = requests.utils.requote_uri(query)
        url = f"https://news.google.com/rss/search?q={q}&hl=en-{country}&gl={country}&ceid={country}:en"
        feed = feedparser.parse(url)
        items = []
        for ent in feed.entries[:max_items]:
            # normalize
            published = ent.get("published") or ent.get("published_parsed")
            items.append({
                "title": ent.get("title"),
                "description": ent.get("summary"),
                "url": ent.get("link"),
                "source": (ent.get("source") or {}).get("title") if ent.get("source") else None,
                "publishedAt": published
            })
        return items
    except Exception:
        return []

@st.cache_data(ttl=CACHE_TTL)
def fetch_newsapi(query: str, max_items=10):
    """Fetch from NewsAPI if key present"""
    if not NEWSAPI_KEY:
        return None
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "pageSize": max_items, "sortBy": "publishedAt", "language": "en", "apiKey": NEWSAPI_KEY}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        js = r.json()
        items = []
        for a in js.get("articles", []):
            items.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "source": a.get("source", {}).get("name"),
                "publishedAt": a.get("publishedAt")
            })
        return items
    except Exception:
        return None

@st.cache_data(ttl=CACHE_TTL)
def fetch_index_latest(sym: str):
    """Return latest close and pct change for an index or ticker via yfinance safely"""
    try:
        df = yf.download(sym, period="5d", interval="1d", progress=False, threads=False)
        if df is None or df.empty:
            return None
        df = df.reset_index()
        last = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2]) if len(df) > 1 else last
        pct = (last - prev) / prev * 100 if prev != 0 else 0.0
        return {"last": last, "pct": pct}
    except Exception:
        return None

@st.cache_data(ttl=CACHE_TTL)
def fetch_stock_history(sym: str, period="1y", interval="1d"):
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={"Close":"close"})
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL)
def fetch_stock_info(sym: str):
    try:
        tk = yf.Ticker(sym)
        info = {}
        # fast_info exists on newer yfinance; fallback to info
        try:
            fast = tk.fast_info
            info.update(fast if isinstance(fast, dict) else {})
        except Exception:
            pass
        try:
            raw = tk.info
            if isinstance(raw, dict):
                info.update(raw)
        except Exception:
            pass
        # dividends/splits
        try:
            divs = tk.dividends
            splits = tk.splits
            info["dividends"] = divs.tail(10).to_dict() if (hasattr(divs, "empty") and not divs.empty) else {}
            info["splits"] = splits.tail(10).to_dict() if (hasattr(splits, "empty") and not splits.empty) else {}
        except Exception:
            info["dividends"] = {}
            info["splits"] = {}
        return info
    except Exception:
        return {}

# Simple trending extractor
def trending_terms(headlines, top_n=6):
    stop = set(["the","and","for","with","from","that","this","are","was","will","have","has","india"])
    words = []
    for h in headlines:
        if not h: continue
        for w in h.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if len(w) > 3 and w not in stop:
                words.append(w)
    return [w for w,_ in Counter(words).most_common(top_n)]

# -----------------------------
# ---- MOSPI / DATA.GOV API ----
# -----------------------------
# We'll try options: TradingEconomics (if key), data.gov.in (if resource id + API key), else fallback to MOSPI scrape or upload

@st.cache_data(ttl=3600)
def fetch_data_gov_resource(resource_id: str, limit: int = 100):
    """Call data.gov.in resource API if DATA_GOV_API_KEY present and resource_id provided."""
    if not resource_id:
        return None
    if not DATA_GOV_API_KEY:
        return None
    base = "https://api.data.gov.in/resource/" + resource_id
    params = {"api-key": DATA_GOV_API_KEY, "limit": limit}
    try:
        r = requests.get(base, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        records = js.get("records") or []
        return pd.DataFrame(records)
    except Exception:
        return None

@st.cache_data(ttl=3600)
def mospi_cpi_scrape():
    """Best-effort MOSPI scrape for CPI if no API resource available."""
    try:
        url = "https://new.mospi.gov.in/dashboard/cpi"
        r = requests.get(url, timeout=15, headers={"User-Agent":"Mozilla/5.0"})
        r.raise_for_status()
        # Try to find csv links
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".csv"):
                if href.startswith("/"):
                    href = "https://new.mospi.gov.in" + href
                rr = requests.get(href, timeout=15)
                if rr.status_code == 200:
                    try:
                        return pd.read_csv(pd.io.common.BytesIO(rr.content))
                    except Exception:
                        continue
        return None
    except Exception:
        return None

# Wrapper for getting CPI/IIP/GDP DF
def get_macro_df(kind: str):
    """
    kind: 'cpi', 'iip', 'gdp'
    Strategy:
     - Try data.gov with provided RESOURCE_ID
     - Else try tradingeconomics (if key)
     - Else try MOSPI scrape (for cpi)
     - Else return None
    """
    if kind == "cpi":
        rid = CPI_RESOURCE_ID
    elif kind == "iip":
        rid = IIP_RESOURCE_ID
    elif kind == "gdp":
        rid = GDP_RESOURCE_ID
    else:
        rid = ""

    # try data.gov
    if rid:
        df = fetch_data_gov_resource(rid, limit=5000)
        if df is not None and not df.empty:
            return df

    # try tradingeconomics (if key)
    if TRADINGECONOMICS_KEY:
        try:
            base = "https://api.tradingeconomics.com/historical/country/india/indicator/"
            # indicator names might vary; map
            mapping = {"cpi":"inflation rate","iip":"index of industrial production","gdp":"gdp"}
            indicator = mapping.get(kind, "")
            if indicator:
                r = requests.get(base + indicator, params={"c": TRADINGECONOMICS_KEY}, timeout=15)
                if r.status_code == 200:
                    js = r.json()
                    # was list of {Date:..., Value:...}
                    return pd.DataFrame(js)
        except Exception:
            pass

    # try MOSPI scrape for cpi
    if kind == "cpi":
        df = mospi_cpi_scrape()
        if df is not None and not df.empty:
            return df

    return None

# -----------------------------
# ---------- UI ---------------
# -----------------------------

# inject CSS for palette + card styles
st.markdown(f"""
<style>
:root {{
  --bg1: {PALETTE['bg1']};
  --bg2: {PALETTE['bg2']};
  --navy: {PALETTE['navy']};
  --teal: {PALETTE['teal']};
  --pos: {PALETTE['pos']};
  --neg: {PALETTE['neg']};
  --neu: {PALETTE['neu']};
  --white: {PALETTE['white']};
}}
body {{ background: linear-gradient(180deg, var(--bg1), var(--bg2)); }}
.stApp {{ background: transparent; }}
.card {{ background: rgba(255,255,255,0.95); border-radius:10px; padding:12px; margin-bottom:10px; color:var(--navy); box-shadow:0 6px 18px rgba(0,0,0,0.06); }}
.headline {{ font-weight:600; color:var(--navy); font-size:16px; }}
.meta {{ color:var(--teal); font-size:12px; }}
.sent-badge {{ padding:4px 8px; border-radius:10px; color:white; font-weight:600; font-size:12px; }}
.index-tile {{ padding:12px; background: rgba(255,255,255,0.95); border-radius:8px; text-align:center; color:var(--navy); }}
</style>
""", unsafe_allow_html=True)

# Header area
st.markdown(f"<h1 style='color:{PALETTE['navy']}'>📰 News & Insights — Economic Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<div style='color:{PALETTE['teal']}; margin-top:-10px; margin-bottom:8px'>Live economic, policy, infrastructure, employment, markets & corporate news — personalised feed</div>", unsafe_allow_html=True)

# Controls row
control_col, indices_col = st.columns([3,1])
with control_col:
    query = st.text_input("Search (enter keywords, OR-separated):", value="India economy OR RBI OR MOSPI OR inflation OR GDP OR employment OR infrastructure OR stock market")
    headlines_count = st.slider("Headlines to show", 3, 20, 8)
    interests = st.multiselect("Your interests (helps personalize)", options=["RBI","infrastructure","startups","banks","inflation","GDP","employment","policy"], default=["RBI","infrastructure"])
    search_stock = st.text_input("Stock symbol (for per-stock view):", value="RELIANCE.NS")
    refresh_manual = st.button("Refresh now")

with indices_col:
    st.markdown("<div class='card'><b>Indices Snapshot</b></div>", unsafe_allow_html=True)
    # show small tiles for indices
    idx_tiles = st.columns(len(INDICES))
    for i,(name,ticker) in enumerate(INDICES.items()):
        res = fetch_index_latest(ticker)
        with idx_tiles[i]:
            if res:
                arrow = "▲" if res["pct"]>=0 else "▼"
                color = PALETTE["pos"] if res["pct"]>=0 else PALETTE["neg"]
                st.markdown(f"<div class='index-tile'><div style='font-weight:700'>{name}</div><div style='font-size:18px;margin-top:6px'>{res['last']:, .2f}</div><div style='color:{color}; font-weight:700'>{arrow} {res['pct']:+.2f}%</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='index-tile'><div style='font-weight:700'>{name}</div><div style='color:#777'>N/A</div></div>", unsafe_allow_html=True)

st.markdown("---")

# Auto-refresh (use st_autorefresh if available)
auto_options = {"Off":0, "30s":30, "1m":60, "5m":300}
auto_choice = st.selectbox("Auto-refresh interval (page reload)", options=list(auto_options.keys()), index=2)
auto_seconds = auto_options[auto_choice]
if HAS_ST_AUTOREFRESH and auto_seconds>0:
    # st_autorefresh returns the number of times rerun happened (int) — useful to trigger reloads
    st_autorefresh(interval=auto_seconds*1000, key="autorefresh")

# Fetch news: prefer NewsAPI else fallback to Google RSS
with st.spinner("Fetching news..."):
    news_items = None
    if NEWSAPI_KEY:
        news_items = fetch_newsapi(query, max_items=headlines_count)
    if not news_items:
        news_items = fetch_google_rss(query, max_items=headlines_count)

# Refresh manual: clear caches and refetch
if refresh_manual:
    st.cache_data.clear()
    if NEWSAPI_KEY:
        news_items = fetch_newsapi(query, max_items=headlines_count)
    news_items = news_items or fetch_google_rss(query, max_items=headlines_count)

if not news_items:
    st.warning("No news found for this query. Try broader keywords or check NEWSAPI_KEY in environment.")
    st.stop()

# Build DF
rows = []
for it in news_items:
    title = it.get("title") or ""
    desc = it.get("description") or ""
    url = it.get("url") or ""
    src = it.get("source") or ""
    pub = it.get("publishedAt") or it.get("published")
    label, score = classify_sentiment(title + ". " + desc)
    rows.append({"title":title,"description":desc,"url":url,"source":src,"publishedAt":pub,"sent_label":label,"sent_score":score})
df = pd.DataFrame(rows)

# Trending & personalization
top_trending = trending_terms(df["title"].tolist(), top_n=6)
st.markdown("### 🔎 Trending & Personalized")
tleft, tright = st.columns([2,3])
with tleft:
    st.write("Trending terms:")
    st.write(", ".join(top_trending) if top_trending else "—")
with tright:
    st.write("Top picks for you (based on interests):")
    def score_interest(r):
        txt = (r["title"]+" "+(r["description"] or "")).lower()
        s = 0
        for it in interests:
            if it.lower() in txt: s += 1
        # trending boost
        for t in top_trending:
            if t in txt: s += 0.5
        return s
    df["interest_score"] = df.apply(score_interest, axis=1)
    top_for_you = df.sort_values(["interest_score","sent_score"], ascending=False).head(3)
    if top_for_you.empty:
        st.write("No personalized picks for this moment.")
    else:
        for _,r in top_for_you.iterrows():
            st.markdown(f"- **{r['title']}**  —  <span style='color:{PALETTE['teal']}'>{r['source']}</span>", unsafe_allow_html=True)

st.markdown("---")

# Headlines list (compact with inline sentiment badge)
st.markdown("### 🗞 Headlines")
for _, r in df.iterrows():
    color = PALETTE["pos"] if r["sent_label"]=="positive" else (PALETTE["neg"] if r["sent_label"]=="negative" else PALETTE["neu"])
    badge = f"<span class='sent-badge' style='background:{color}'>{r['sent_label'].upper()}</span>"
    pub = r["publishedAt"] or ""
    st.markdown(f"<div class='card'><div class='headline'><a href='{r['url']}' target='_blank' style='color:{PALETTE['navy']}'>{r['title']}</a> {badge}</div>", unsafe_allow_html=True)
    if r["description"]:
        st.markdown(f"<div class='meta'>{r['description']}</div>", unsafe_allow_html=True)
    if r["source"] or pub:
        st.markdown(f"<div style='color:{PALETTE['teal']}; font-size:12px; margin-top:6px'>{r['source']} · {r['publishedAt']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# MOSPI / Macro micro-charts
# -----------------------------
st.markdown("---")
st.markdown("## 📈 Macro micro-charts (CPI / IIP / GDP) — MOSPI / Data.gov fallback")

macro_cols = st.columns(3)
kinds = ["cpi","iip","gdp"]
for col, kind in zip(macro_cols, kinds):
    with col:
        st.markdown(f"### {kind.upper()}")
        df_macro = get_macro_df(kind)
        if df_macro is None or df_macro.empty:
            st.info(f"No {kind.upper()} data found automatically. You can upload a CSV/XLSX file below.")
            uploaded = st.file_uploader(f"Upload {kind.upper()} CSV/XLSX (optional)", type=["csv","xlsx"], key=f"up_{kind}")
            if uploaded:
                try:
                    if uploaded.name.endswith(".csv"):
                        df_user = pd.read_csv(uploaded)
                    else:
                        df_user = pd.read_excel(uploaded)
                    st.dataframe(df_user.head())
                    # attempt to find numeric column and date
                    date_cols = [c for c in df_user.columns if 'date' in c.lower() or 'month' in c.lower()]
                    val_cols = [c for c in df_user.columns if any(x in c.lower() for x in ['value','index','cpi','iip','gdp'])]
                    if not date_cols:
                        date_cols = [df_user.columns[0]]
                    if date_cols and val_cols:
                        df_user['__date'] = pd.to_datetime(df_user[date_cols[0]], errors='coerce')
                        fig = px.line(df_user.sort_values('__date'), x='__date', y=val_cols[0], title=f"{kind.upper()} (uploaded)")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error("Upload parse error: "+str(e))
        else:
            # try to autodetect date & value columns
            # For tradingeconomics result, look for 'Date' or 'date' and 'Value' or 'value'
            cols = df_macro.columns.tolist()
            date_col = next((c for c in cols if 'date' in c.lower()), cols[0])
            val_col = next((c for c in cols if any(x in c.lower() for x in ['value','index','cpi','iip','gdp','amount','price'])), None)
            try:
                df_macro[date_col] = pd.to_datetime(df_macro[date_col], errors='coerce')
            except Exception:
                pass
            if val_col:
                fig = px.line(df_macro.sort_values(date_col), x=date_col, y=val_col, title=f"{kind.upper()} (latest)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(df_macro.head())

# -----------------------------
# Per-stock section (single symbol)
# -----------------------------
st.markdown("---")
st.markdown("## 💹 Stock — single symbol search & corporate actions")
stock_col, stock_meta_col = st.columns([3,1])
with stock_col:
    sym = st.text_input("Enter stock symbol (e.g., RELIANCE.NS or AAPL)", value=search_stock, key="stock_input")
    if sym:
        st.markdown(f"### {sym}")
        hist = fetch_stock_history(sym, period="1y")
        if hist.empty:
            st.warning("No price history for this symbol.")
        else:
            latest = hist["close"].iloc[-1]
            prev = hist["close"].iloc[-2] if len(hist) > 1 else latest
            pct = (latest - prev) / prev * 100 if prev != 0 else 0.0
            color = PALETTE["pos"] if pct >= 0 else PALETTE["neg"]
            st.metric(sym, f"{latest:,.2f}", f"{pct:+.2f}%")
            fig = px.line(hist, x="Date", y="close", title=f"{sym} — 1 year", labels={"close":"Close"})
            fig.update_traces(line=dict(color=color))
            fig.update_layout(paper_bgcolor=PALETTE["bg2"], plot_bgcolor=PALETTE["bg2"])
            st.plotly_chart(fig, use_container_width=True)

            # corporate actions
            info = fetch_stock_info(sym)
            st.markdown("### Corporate actions")
            divs = info.get("dividends", {}) if isinstance(info, dict) else {}
            splits = info.get("splits", {}) if isinstance(info, dict) else {}
            if divs:
                try:
                    ddf = pd.DataFrame(list(divs.items()), columns=["Date","Dividend"])
                    ddf["Date"] = pd.to_datetime(ddf["Date"], errors='coerce')
                    st.table(ddf.sort_values("Date", ascending=False).head(8))
                except Exception:
                    st.write(divs)
            else:
                st.write("No dividend data (via yfinance).")
            if splits:
                try:
                    sdf = pd.DataFrame(list(splits.items()), columns=["Date","Split"])
                    sdf["Date"] = pd.to_datetime(sdf["Date"], errors='coerce')
                    st.table(sdf.sort_values("Date", ascending=False).head(8))
                except Exception:
                    st.write(splits)
            else:
                st.write("No splits data (via yfinance).")
with stock_meta_col:
    st.markdown("## Related news for stock")
    related = None
    # try NewsAPI first
    if NEWSAPI_KEY:
        related = fetch_newsapi(sym, max_items=6)
    if not related:
        related = fetch_google_rss(sym + " stock", max_items=6)
    if not related:
        st.info("No related news found.")
    else:
        for it in related:
            t = it.get("title")
            u = it.get("url")
            lab, sc = classify_sentiment(t + " " + (it.get("description") or ""))
            badge_color = PALETTE["pos"] if lab=="positive" else (PALETTE["neg"] if lab=="negative" else PALETTE["neu"])
            st.markdown(f"- <a href='{u}' target='_blank'>{t}</a> <span style='color:{badge_color}'>({lab})</span>", unsafe_allow_html=True)

# -----------------------------
# Newsletter auto-summary (3-4 bullets)
# -----------------------------
st.markdown("---")
st.markdown("## 📝 Auto Newsletter — short brief (3–4 bullets)")
with st.spinner("Generating brief..."):
    # pick top 3 headlines by interest_score then sentiment
    top_news = df.sort_values(["interest_score","sent_score"], ascending=False).head(4)
    bullets = []
    # include 1 macro bullet if available
    macro_bullet = None
    try:
        # use CPI if df exists
        cpi_df = get_macro_df("cpi")
        if cpi_df is not None and not cpi_df.empty:
            # try to get latest numeric
            cols = cpi_df.columns.tolist()
            valcol = next((c for c in cols if any(x in c.lower() for x in ["value","index","cpi"])), None)
            datecol = next((c for c in cols if "date" in c.lower()), cols[0])
            if valcol:
                latest_row = cpi_df.sort_values(datecol).iloc[-1]
                macro_bullet = f"CPI update: latest {valcol} = {latest_row[valcol]} ({format_time(latest_row.get(datecol))})"
    except Exception:
        macro_bullet = None

    if macro_bullet:
        bullets.append(macro_bullet)

    for _, r in top_news.iterrows():
        short = (r["title"] if len(r["title"])<140 else r["title"][:137]+"...")
        tag = r["sent_label"].capitalize()
        bullets.append(f"{short} — {tag}")

    # ensure 3-4 bullets
    bullets = bullets[:4] if len(bullets)>4 else bullets
    if not bullets:
        st.write("Not enough items to create newsletter.")
    else:
        newsletter_text = "Daily Economic Brief — Auto-generated\n\n"
        for i,b in enumerate(bullets, start=1):
            newsletter_text += f"{i}. {b}\n\n"
        st.text_area("Newsletter (editable)", value=newsletter_text, height=220)
        st.download_button("Download newsletter (TXT)", data=newsletter_text, file_name="economic_brief.txt")

# Footer
st.markdown("---")
st.markdown(f"<div style='color:{PALETTE['teal']}'>Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} • Cached for {CACHE_TTL} seconds where applicable</div>", unsafe_allow_html=True)
