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
from textblob import TextBlob

import streamlit as st

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
MARKET_TTL = 60
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
    params = {"q": query, "language": "en", "pageSize": n, "sortBy": "publishedAt", "apiKey": NEWSAPI_KEY}
    try:
        js = safe_json_get(url, params=params)
        if js and js.get("status") == "ok":
            out = []
            for a in js.get("articles", [])[:n]:
                out.append({
                    "title": a.get("title"),
                    "summary": a.get("description") or a.get("content") or "",
                    "url": a.get("url"),
                    "source": a.get("source", {}).get("name"),
                    "publishedAt": a.get("publishedAt")
                })
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
            out.append({
                "title": entry.get("title"),
                "summary": entry.get("summary") or "",
                "url": entry.get("link"),
                "source": (entry.get("source") or {}).get("title") if entry.get("source") else None,
                "publishedAt": entry.get("published") or entry.get("published_parsed")
            })
        return out
    except Exception as e:
        log(f"google rss error: {e}")
        return []

def fetch_news(query, n=8):
    res = fetch_newsapi(query, n=n) if NEWSAPI_KEY else None
    if res:
        return res
    return fetch_google_rss(query, n=n)

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
    try:
        t = yf.Ticker(sym)
        divs = t.dividends if hasattr(t, "dividends") else pd.Series(dtype=float)
        splits = t.splits if hasattr(t, "splits") else pd.Series(dtype=float)
        news = []
        try:
            raw = t.news
            if isinstance(raw, list):
                for item in raw[:8]:
                    news.append({"title": item.get("title"), "link": item.get("link")})
        except Exception:
            pass
        return {"dividends": divs, "splits": splits, "news": news}
    except Exception as e:
        log(f"stock actions error {sym}: {e}")
        return {"dividends": pd.Series(dtype=float), "splits": pd.Series(dtype=float), "news": []}

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
search_query = st.sidebar.text_input("Search query", value="India economy OR RBI OR MOSPI OR inflation OR GDP OR infrastructure")
headlines_count = st.sidebar.slider("Headlines to show", min_value=3, max_value=20, value=6)
auto_ref = st.sidebar.selectbox("Auto-refresh", options=["Off","30s","1m","5m"], index=2)
stock_input = st.sidebar.text_input("Single stock (one symbol)", value="RELIANCE.NS")
st.sidebar.markdown("---")
st.sidebar.markdown("**Interests (for personalization)**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Interests (for personalization)**")

options_list = ["RBI", "infrastructure", "startups", "banks", "inflation", "GDP", "employment", "policy", "stock"]
saved_prefs = [p for p in st.session_state.get("prefs", []) if p in options_list]

prefs = st.sidebar.multiselect(
    "Pick interests",
    options=options_list,
    default=saved_prefs
)

# ✅ FIXED INDENTATION HERE
if st.sidebar.button("Save interests"):
    st.session_state["prefs"] = prefs
st.sidebar.markdown("---")
st.sidebar.markdown("Advanced: put keys in .env or Streamlit secrets (NEWSAPI_KEY, DATA_GOV_API_KEY, CPI_RESOURCE_ID, IIP_RESOURCE_ID, GDP_RESOURCE_ID)")
if st.sidebar.button("Refresh now"):
    requests_cache.clear()
    st.experimental_rerun()

# parse auto_ref seconds
interval_map = {"Off":0,"30s":30,"1m":60,"5m":300}
interval_seconds = interval_map.get(auto_ref, 0)
if HAS_AUTOREF and interval_seconds > 0:
    tick = st_autorefresh(interval=interval_seconds*1000, key="autorefresh_counter")
    st.sidebar.caption(f"Auto-refresh ticks: {tick}")
elif interval_seconds > 0:
    st.sidebar.info("Auto-refresh set; install streamlit-autorefresh for automatic reloads.")

# ---------- Top header & indices ----------
st.markdown("<h1>📰 News & Insights — India Economic Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Live economic, policy, infrastructure, employment, markets & corporate news — personalized feed</div>", unsafe_allow_html=True)
st.markdown("---")

with st.spinner("Fetching market snapshot..."):
    indices = fetch_index_snapshot()

# indices tiles
cols = st.columns(len(INDICES))
for i,(name,sym) in enumerate(INDICES.items()):
    val = indices.get(name, {})
    if val.get("last") is None:
        cols[i].markdown(f"<div class='card'><b>{name}</b><div class='small-muted'>N/A</div></div>", unsafe_allow_html=True)
    else:
        color = PALETTE["pos"] if val["pct"] >= 0 else PALETTE["neg"]
        arrow = "▲" if val["pct"] >= 0 else "▼"
        cols[i].markdown(f"<div class='card'><b>{name}</b><div style='font-size:18px'>{val['last']:,.2f}</div><div style='color:{color}; font-weight:700'>{arrow} {val['pct']:+.2f}%</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Fetch news ----------
with st.spinner("Fetching news..."):
    raw_news = fetch_news(search_query, n=headlines_count)
    if not raw_news:
        st.info("No news found for this query (NewsAPI may be required). Using broader search.")
        raw_news = fetch_news(search_query.split(" ")[0], n=headlines_count)  # fallback attempt

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
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context) as smtp:
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
            st.success("Newsletter sent.")
        except Exception as e:
            st.error(f"Send failed: {e}")
else:
    st.info("SMTP not configured. To enable send, set SMTP_* env vars.")

# ---------- MOSPI / Macro micro-charts ----------
st.markdown("---")
st.markdown("## 📈 MOSPI / Macro Indicators (CPI / IIP / GDP) — auto or upload fallback")
st.markdown("If automatic fetch fails, upload CSV/XLSX with clear date + value columns.")

def load_macro(kind, resource_id):
    uploaded = st.file_uploader(f"Upload {kind.upper()} CSV/XLSX (fallback)", type=["csv","xlsx"], key=f"up_{kind}")
    df = None; source = None
    if resource_id and DATA_GOV_API_KEY:
        j = fetch_data_gov_resource(resource_id, limit=2000)
        if j and j.get("records"):
            df = pd.DataFrame(j["records"])
            source = f"data.gov resource {resource_id}"
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

def auto_plot(df, label):
    if df is None:
        st.info(f"No {label} data available to auto-plot.")
        return
    cols = df.columns.tolist()
    date_col = next((c for c in cols if "date" in c.lower() or "month" in c.lower()), None)
    val_col = next((c for c in cols if any(x in c.lower() for x in ["value","index","cpi","iip","gdp","amount","price"])), None)
    if date_col and val_col:
        try:
            tmp = df[[date_col, val_col]].copy()
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            tmp = tmp.dropna(subset=[date_col, val_col]).sort_values(date_col)
            fig = px.line(tmp, x=date_col, y=val_col, title=f"{label} — {val_col}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plot error {label}: {e}")
    else:
        st.dataframe(df.head())
        st.info("Could not auto-detect date/value columns. Upload CSV with 'date' and 'value' naming for auto-plot.")

mcols = st.columns(3)
with mcols[0]:
    st.markdown("### CPI")
    if cpi_src: st.caption(cpi_src)
    auto_plot(cpi_df, "CPI")
with mcols[1]:
    st.markdown("### IIP")
    if iip_src: st.caption(iip_src)
    auto_plot(iip_df, "IIP")
with mcols[2]:
    st.markdown("### GDP")
    if gdp_src: st.caption(gdp_src)
    auto_plot(gdp_df, "GDP")

# ---------- Single-stock deep dive ----------
st.markdown("---")
st.markdown("## 💹 Stock — single symbol deep dive (chart + corporate actions + related news)")
st.markdown("Enter symbol in sidebar (e.g., RELIANCE.NS).")

if stock_input:
    with st.spinner(f"Fetching {stock_input} ..."):
        sh = fetch_stock_history(stock_input, period="1y")
        sa = fetch_stock_actions(stock_input)
    if sh.empty:
        st.warning("No history for this symbol. Check symbol suffix (.NS for NSE).")
    else:
        latest = sh["close"].iloc[-1]
        prev = sh["close"].iloc[-2] if len(sh) > 1 else latest
  # --- Safe percentage change calculation ---
try:
    if prev is None or pd.isna(prev) or prev == 0:
        pct = 0.0
    else:
        pct = ((latest - prev) / prev) * 100
except Exception:
    pct = 0.0

# --- Stock performance metric (safe casting) ---
try:
    latest_val = float(latest) if pd.notna(latest) else 0.0
    pct_val = float(pct) if pd.notna(pct) else 0.0
    st.metric(f"{stock_input} Latest", f"{latest_val:,.2f}", f"{pct_val:+.2f}%")
except Exception as e:
    st.warning(f"Could not display metric for {stock_input}: {e}")
# --- Stock chart (safe rendering) ---
if sh is not None and not sh.empty:
    if "Date" not in sh.columns:
        sh = sh.reset_index()  # Ensure Date column exists for Plotly
try:
    fig = px.line(
        sh,
        x="Date",
        y="close",
        title=f"{stock_input} – 1 year",
        labels={"close": "Price", "Date": "Date"},
    )
    fig.update_traces(line=dict(color=PALETTE["pos"] if pct >= 0 else PALETTE["neg"], width=2))
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Chart rendering failed for {stock_input}: {e}")# --- Moving Averages Overlay ---
st.markdown("### 📊 Moving Averages (Trend Analysis)")

try:
    # Checkbox controls
    show_ma20 = st.checkbox("Show MA20 (Short-term)", value=True)
    show_ma50 = st.checkbox("Show MA50 (Medium-term)", value=True)
    show_ma200 = st.checkbox("Show MA200 (Long-term)", value=False)

    # Calculate moving averages
    sh["MA20"] = sh["close"].rolling(window=20).mean()
    sh["MA50"] = sh["close"].rolling(window=50).mean()
    sh["MA200"] = sh["close"].rolling(window=200).mean()

    # Create chart with moving averages
    fig_ma = px.line(
        sh,
        x="Date",
        y="close",
        title=f"{stock_input.upper()} Trend Overview (with Moving Averages)",
        labels={"close": "Price (₹)", "Date": "Date"},
    )

    # Update the base line (actual stock price)
    fig_ma.update_traces(line=dict(color=PALETTE["pos"] if pct >= 0 else PALETTE["neg"], width=2))

    # Add optional moving average lines
    if show_ma20:
        fig_ma.add_scatter(
            x=sh["Date"], y=sh["MA20"],
            mode="lines", name="MA20 (Short)",
            line=dict(width=1.8, dash="dot", color=PALETTE["teal"])
        )
    if show_ma50:
        fig_ma.add_scatter(
            x=sh["Date"], y=sh["MA50"],
            mode="lines", name="MA50 (Medium)",
            line=dict(width=1.8, dash="dot", color=PALETTE["mid"])
        )
    if show_ma200:
        fig_ma.add_scatter(
            x=sh["Date"], y=sh["MA200"],
            mode="lines", name="MA200 (Long)",
            line=dict(width=1.8, dash="dot", color=PALETTE["neg"])
        )

    st.plotly_chart(fig_ma, use_container_width=True)

except Exception as e:
    st.warning(f"Moving average overlay unavailable: {e}")
    show_ma20 = st.checkbox("Show MA20 (Short-term)", value=True)
    show_ma50 = st.checkbox("Show MA50 (Medium-term)", value=True)
    show_ma200 = st.checkbox("Show MA200 (Long-term)", value=False)

    # Calculate moving averages
    sh["MA20"] = sh["close"].rolling(window=20).mean()
    sh["MA50"] = sh["close"].rolling(window=50).mean()
    sh["MA200"] = sh["close"].rolling(window=200).mean()

    # Create chart with moving averages
    fig_ma = px.line(
        sh,
        x="Date",
        y="close",
        title=f"{stock_input.upper()} Trend Overview (with Moving Averages)",
        labels={"close": "Price (₹)", "Date": "Date"},
    )

    # Update the base line (actual stock price)
    fig_ma.update_traces(line=dict(color=PALETTE["pos"] if pct >= 0 else PALETTE["neg"], width=2))

    # Add optional moving average lines
    if show_ma20:
        fig_ma.add_scatter(
            x=sh["Date"], y=sh["MA20"],
            mode="lines", name="MA20 (Short)",
            line=dict(width=1.8, dash="dot", color=PALETTE["teal"])
        )
    if show_ma50:
        fig_ma.add_scatter(
# --- Moving Averages Overlay ---
try:
    st.markdown("### 📊 Moving Averages (Trend Analysis)")

    # Checkbox controls
    show_ma20 = st.checkbox("Show MA20 (Short-term)", value=True)
    show_ma50 = st.checkbox("Show MA50 (Medium-term)", value=True)
    show_ma200 = st.checkbox("Show MA200 (Long-term)", value=False)

    # Calculate moving averages
    sh["MA20"] = sh["close"].rolling(window=20).mean()
    sh["MA50"] = sh["close"].rolling(window=50).mean()
    sh["MA200"] = sh["close"].rolling(window=200).mean()

    # Create chart with moving averages
    fig_ma = px.line(
        sh,
        x="Date",
        y="close",
        title=f"{stock_input.upper()} Trend Overview (with Moving Averages)",
        labels={"close": "Price (₹)", "Date": "Date"},
    )

    if show_ma20:
        fig_ma.add_scatter(
            x=sh["Date"], y=sh["MA20"],
            mode="lines", name="MA20 (Short)",
            line=dict(width=1.5, dash="dot", color=PALETTE["pos"])
        )
    if show_ma50:
        fig_ma.add_scatter(
            x=sh["Date"], y=sh["MA50"],
            mode="lines", name="MA50 (Medium)",
            line=dict(width=1.5, dash="dot", color=PALETTE["warn"])
        )
if show_ma20:
    fig_ma.add_scatter(
        x=sh["Date"], y=sh["MA20"],
        mode="lines", name="MA20 (Short)",
        line=dict(width=1.5, dash="dot", color=PALETTE["pos"])
    )

if show_ma50:
    fig_ma.add_scatter(
        x=sh["Date"], y=sh["MA50"],
        mode="lines", name="MA50 (Medium)",
        line=dict(width=1.5, dash="dot", color=PALETTE["warn"])
    )

if show_ma200:
    fig_ma.add_scatter(
        x=sh["Date"], y=sh["MA200"],
        mode="lines", name="MA200 (Long)",
        line=dict(width=1.8, dash="dot", color=PALETTE["neg"])
    )

st.plotly_chart(fig_ma, use_container_width=True)
except Exception as e:
    st.warning(f"Moving average overlay unavailable: {e}")

else:
    st.warning(
        f"No historical data found for {stock_input}. "
        f"Check symbol (e.g., RELIANCE.NS for NSE)."
    )

# --- Corporate actions ---
st.markdown("### Corporate actions")
# --- Corporate actions ---
st.markdown("### Corporate actions")
except Exception as e:
    st.warning(f"Moving average overlay unavailable: {e}")

else:
    st.warning(
        f"No historical data found for {stock_input}. "
        f"Check symbol (e.g., RELIANCE.NS for NSE)."
    )

# --- Corporate actions ---
st.markdown("### Corporate actions")
try:
    divs = sa.get("dividends")
    splits = sa.get("splits")

    if not getattr(divs, "empty", True):
        ddf = divs.reset_index().rename(columns={"Date": "Date", 0: "Dividend"}) if isinstance(divs, pd.Series) else divs
        st.dataframe(ddf.tail(8))
    else:
        st.info("No dividends found (yfinance).")

    if not getattr(splits, "empty", True):
        sdf = splits.reset_index().rename(columns={"Date": "Date", 0: "Split"}) if isinstance(splits, pd.Series) else splits
        st.dataframe(sdf.tail(8))
    else:
        st.info("No splits found (yfinance).")

except Exception as e:
    st.error(f"Corporate actions error: {e}")

# Related news
st.markdown("### Related news (search fallback)")
related = fetch_news(f"{search_query} {stock_input}", n=6)
if related:
    for r in related:
        st.markdown(f"- <a href='{r.get('url')}' target='_blank'>{r.get('title')}</a>", unsafe_allow_html=True)
else:
    st.info("No related news found.")
# ---------- Footer & debug ----------
st.markdown("---")
st.markdown(f"<div style='color:{PALETTE['teal']}'>Last update: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>", unsafe_allow_html=True)

with st.expander("Show internal debug log"):
    for r in st.session_state["_log"][-200:]:
        st.text(r)

st.markdown("---")
st.markdown("**Tips:** set NEWSAPI_KEY and DATA_GOV_API_KEY in .env or Streamlit Secrets for better data. Upload CSVs if automatic fetch fails.")
