import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import datetime
import time
import requests
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from bs4 import BeautifulSoup

# ─── Load API Key (Streamlit Cloud Secrets → .env fallback) ──
load_dotenv()

# Try Streamlit Cloud secrets first (for deployed app), then .env (for local)
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ─── Initialize Gemini Client ────────────────────────────
gemini_client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "PASTE_YOUR_GEMINI_API_KEY_HERE":
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ─── Fix Yahoo Finance Rate Limiting on Cloud ────────────
# Create a custom session with browser-like headers to avoid IP blocks
_yf_session = requests.Session()
_yf_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
})

# ─── Page Config ──────────────────────────────────────────
st.set_page_config(page_title="AI Stock Analyzer", layout="wide", page_icon="📈")

# ─── Custom CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

body, .stApp { 
    background-color: #0a0a1a; 
    font-family: 'Inter', sans-serif; 
}
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; border: none;
    border-radius: 25px; font-weight: 600; width: 100%;
    padding: 0.6rem 1.2rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(102, 126, 234, 0.5);
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px; padding: 15px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
[data-testid="stMetricValue"] { color: #00d4ff; }
[data-testid="stMetricLabel"] { color: #8892b0; }
.agent-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #0d1b2a 100%);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 16px; padding: 20px; margin: 10px 0;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.agent-step {
    background: rgba(102, 126, 234, 0.08);
    border-left: 3px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin: 8px 0;
    color: #c0c0d0; font-size: 0.9rem;
}
.signal-buy {
    background: linear-gradient(135deg, #00b09b, #96c93d);
    color: white; padding: 8px 20px; border-radius: 20px;
    display: inline-block; font-weight: 700; font-size: 1.1rem;
}
.signal-sell {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white; padding: 8px 20px; border-radius: 20px;
    display: inline-block; font-weight: 700; font-size: 1.1rem;
}
.signal-hold {
    background: linear-gradient(135deg, #f7971e, #ffd200);
    color: #1a1a2e; padding: 8px 20px; border-radius: 20px;
    display: inline-block; font-weight: 700; font-size: 1.1rem;
}
h1, h2, h3 { color: #e0e0ff !important; }
.stExpander { 
    background: #1a1a2e; 
    border: 1px solid rgba(102, 126, 234, 0.2); 
    border-radius: 12px; 
}
.powered-badge {
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 20px; padding: 6px 16px;
    display: inline-block; margin-top: 5px;
    font-size: 0.85rem; color: #8892b0;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────
st.markdown("""
<div style="text-align: center; padding: 10px 0 20px 0;">
    <h1 style="background: linear-gradient(135deg, #667eea, #764ba2, #00d4ff); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               font-size: 2.8rem; font-weight: 700; margin-bottom: 0;">
        📈 AI Stock Analyzer
    </h1>
    <p style="color: #8892b0; font-size: 1.1rem; margin-top: 5px;">
        Type any company name — get full AI-powered analysis with agentic research
    </p>
    <span class="powered-badge">⚡ Powered by Google Gemini AI — Free & Instant</span>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0;">
        <h2 style="color: #667eea;">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)

    company_name = st.text_input(
        "🏢 Company Name", 
        placeholder="e.g. Adani, Tesla, Infosys, Apple"
    )
    start_date = st.date_input("📅 Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.date_input("📅 End Date", value=pd.to_datetime("2024-12-31"))

    st.markdown("---")
    st.markdown("##### 🤖 Agentic AI Mode")
    enable_agent = st.toggle("Enable Deep Research Agent", value=True)
    if enable_agent:
        st.caption("AI will autonomously research news, financials, and generate a deep investment report.")

    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze", use_container_width=True)

    # Show API status
    if gemini_client:
        st.success("✅ AI Engine: Connected")
    else:
        st.error("❌ AI Engine: Not configured")
        st.caption("Owner needs to set GEMINI_API_KEY in .env file")

    st.markdown("""
    <div style="margin-top: 30px; padding: 15px; background: rgba(102,126,234,0.1); 
                border-radius: 12px; border: 1px solid rgba(102,126,234,0.2);">
        <p style="color: #8892b0; font-size: 0.8rem; margin: 0;">
            <strong style="color: #667eea;">How it works:</strong><br>
            1. Enter any company name<br>
            2. App auto-finds the stock ticker<br>
            3. Fetches live market data<br>
            4. AI generates deep analysis<br>
            5. Agentic mode researches news & risks<br><br>
            <strong style="color: #00d4ff;">🆓 No API key needed — AI built-in!</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Helper Functions ────────────────────────────────────

def safe_yf_call(func, *args, max_retries=3, default=None, **kwargs):
    """Safely call any yfinance function with retry + exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            if 'RateLimit' in error_str or 'rate' in error_str.lower() or 'Too Many Requests' in error_str:
                wait_time = (2 ** attempt) * 3  # 3s, 6s, 12s
                time.sleep(wait_time)
                continue
            else:
                return default if default is not None else {}
    return default if default is not None else {}


def safe_get_info(ticker_obj, max_retries=3):
    """Safely fetch stock.info with retry + exponential backoff for rate limits."""
    return safe_yf_call(lambda: ticker_obj.info, max_retries=max_retries, default={})


def safe_download(ticker_sym, start, end, max_retries=3):
    """Safely call yf.download with retry logic."""
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker_sym, start=start, end=end, auto_adjust=True, session=_yf_session)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            error_str = f"{type(e).__name__}: {e}"
            if 'RateLimit' in error_str or 'rate' in error_str.lower():
                wait_time = (2 ** attempt) * 3
                time.sleep(wait_time)
                continue
            else:
                return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=300, show_spinner=False)
def get_ticker(company):
    """Auto-find stock ticker from company name using yfinance search."""
    try:
        # Try yfinance Search with retry
        search = None
        for attempt in range(3):
            try:
                search = yf.Search(company, max_results=10)
                break
            except Exception as e:
                if 'RateLimit' in str(type(e).__name__) or 'rate' in str(e).lower():
                    time.sleep((2 ** attempt) * 3)
                    continue
                break

        results = []
        if search:
            results = search.quotes if hasattr(search, 'quotes') else []

        for r in results:
            if r.get('quoteType') in ['EQUITY', 'ETF']:
                return r['symbol'], r.get('shortname', r.get('longname', company))

        # Fallback: try direct ticker lookup
        if not results:
            test = yf.Ticker(company.upper().replace(" ", ""), session=_yf_session)
            fallback_info = safe_get_info(test)
            if fallback_info.get('symbol'):
                return fallback_info['symbol'], fallback_info.get('shortName', company)
        return None, None
    except Exception as e:
        st.warning(f"Search issue: {e}")
        return None, None


def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def safe_squeeze(series_or_df):
    """Safely squeeze a DataFrame column that might have MultiIndex."""
    if isinstance(series_or_df, pd.DataFrame):
        if series_or_df.shape[1] == 1:
            return series_or_df.iloc[:, 0]
        return series_or_df
    return series_or_df


def fetch_news_headlines(company_name, ticker_sym):
    """Fetch recent news headlines for the company."""
    headlines = []
    try:
        stock = yf.Ticker(ticker_sym, session=_yf_session)
        news = stock.news if hasattr(stock, 'news') else []
        if news:
            for item in news[:8]:
                title = item.get('title', '')
                publisher = item.get('publisher', '')
                link = item.get('link', '')
                if title:
                    headlines.append({
                        'title': title,
                        'publisher': publisher,
                        'link': link
                    })
    except Exception:
        pass

    # Fallback: Google News RSS
    if len(headlines) < 3:
        try:
            query = f"{company_name} stock"
            url = f"https://news.google.com/rss/search?q={query}&hl=en&gl=US&ceid=US:en"
            resp = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'xml')
                items = soup.find_all('item', limit=8)
                for item in items:
                    title = item.find('title')
                    source = item.find('source')
                    if title:
                        headlines.append({
                            'title': title.text,
                            'publisher': source.text if source else 'Google News',
                            'link': ''
                        })
        except Exception:
            pass

    return headlines[:8]


@st.cache_data(ttl=300, show_spinner=False)
def get_financial_details(ticker_sym):
    """Fetch deep financial data for agentic analysis."""
    stock = yf.Ticker(ticker_sym, session=_yf_session)
    data = {}

    try:
        info = safe_get_info(stock)
        data['financials'] = {
            'revenue': info.get('totalRevenue', 'N/A'),
            'gross_margins': info.get('grossMargins', 'N/A'),
            'operating_margins': info.get('operatingMargins', 'N/A'),
            'profit_margins': info.get('profitMargins', 'N/A'),
            'debt_to_equity': info.get('debtToEquity', 'N/A'),
            'current_ratio': info.get('currentRatio', 'N/A'),
            'return_on_equity': info.get('returnOnEquity', 'N/A'),
            'return_on_assets': info.get('returnOnAssets', 'N/A'),
            'free_cash_flow': info.get('freeCashflow', 'N/A'),
            'earnings_growth': info.get('earningsGrowth', 'N/A'),
            'revenue_growth': info.get('revenueGrowth', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'peg_ratio': info.get('pegRatio', 'N/A'),
            'book_value': info.get('bookValue', 'N/A'),
            'enterprise_value': info.get('enterpriseValue', 'N/A'),
            'ebitda': info.get('ebitda', 'N/A'),
            'total_cash': info.get('totalCash', 'N/A'),
            'total_debt': info.get('totalDebt', 'N/A'),
            'employees': info.get('fullTimeEmployees', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'website': info.get('website', 'N/A'),
            'summary': info.get('longBusinessSummary', 'N/A'),
        }
    except Exception:
        data['financials'] = {}

    # Get recent earnings
    try:
        earnings = stock.quarterly_earnings
        if earnings is not None and not earnings.empty:
            data['recent_earnings'] = earnings.tail(4).to_dict()
        else:
            data['recent_earnings'] = {}
    except Exception:
        data['recent_earnings'] = {}

    # Get recommendations
    try:
        recs = stock.recommendations
        if recs is not None and not recs.empty:
            data['analyst_recommendations'] = recs.tail(5).to_dict()
        else:
            data['analyst_recommendations'] = {}
    except Exception:
        data['analyst_recommendations'] = {}

    # Get major holders
    try:
        holders = stock.major_holders
        if holders is not None and not holders.empty:
            data['major_holders'] = holders.to_dict()
        else:
            data['major_holders'] = {}
    except Exception:
        data['major_holders'] = {}

    return data


def get_ai_analysis(ticker_sym, full_name, metrics):
    """Standard AI analysis using Google Gemini."""
    prompt = f"""You are a senior equity research analyst. Analyze {full_name} ({ticker_sym}) with these metrics:
- Current Price: {metrics['current_price']}
- 52W High: {metrics['high_52w']}
- 52W Low: {metrics['low_52w']}
- Annualized Volatility: {metrics['volatility']:.2%}
- RSI (14): {metrics['rsi']:.1f}
- PE Ratio: {metrics['pe']}
- Market Cap: {metrics['market_cap']}
- Sector: {metrics['sector']}

Provide a structured analysis with these exact sections:
**Company Overview** — What this company does, its market position (2-3 sentences)
**Technical Analysis** — RSI interpretation, trend analysis, moving average signals
**Buy / Hold / Sell Recommendation** — Clear verdict with reasoning
**Risk Assessment** — Top 3 risks to consider
**Future Outlook (6-12 months)** — Where this stock is headed and why

Be direct, specific, and data-driven. No generic filler."""

    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=1500,
        )
    )
    return response.text


def run_agentic_analysis(ticker_sym, full_name, metrics, news_headlines, financial_details):
    """
    Agentic AI Analysis — Gemini autonomously generates a deep investment report 
    using all gathered data (news, financials, technicals).
    """
    # Format news for the prompt
    news_text = ""
    if news_headlines:
        news_text = "\n".join([
            f"  - {h['title']} ({h['publisher']})" for h in news_headlines
        ])
    else:
        news_text = "  No recent news found."

    # Format financials
    fin = financial_details.get('financials', {})
    fin_text = ""
    if fin:
        fin_items = []
        for key, val in fin.items():
            if val != 'N/A' and val is not None and key != 'summary':
                label = key.replace('_', ' ').title()
                if isinstance(val, (int, float)):
                    if abs(val) >= 1e9:
                        fin_items.append(f"  - {label}: ${val/1e9:.2f}B")
                    elif abs(val) >= 1e6:
                        fin_items.append(f"  - {label}: ${val/1e6:.2f}M")
                    elif abs(val) < 1:
                        fin_items.append(f"  - {label}: {val:.2%}")
                    else:
                        fin_items.append(f"  - {label}: {val:.2f}")
                else:
                    fin_items.append(f"  - {label}: {val}")
        fin_text = "\n".join(fin_items)

    company_summary = fin.get('summary', 'N/A')

    # Analyst recommendations
    recs = financial_details.get('analyst_recommendations', {})
    recs_text = json.dumps(recs, indent=2, default=str) if recs else "No analyst data available."

    # Recent earnings
    earnings = financial_details.get('recent_earnings', {})
    earnings_text = json.dumps(earnings, indent=2, default=str) if earnings else "No recent earnings data."

    agentic_prompt = f"""You are an autonomous AI investment research agent. You have been given the task of producing a 
DEEP investment research report on {full_name} ({ticker_sym}). 

You have autonomously gathered the following data through your research tools:

═══════════════════════════════════════════════════
📊 MARKET DATA (LIVE)
═══════════════════════════════════════════════════
- Current Price: {metrics['current_price']}
- 52-Week High: {metrics['high_52w']}
- 52-Week Low: {metrics['low_52w']}
- Annualized Volatility: {metrics['volatility']:.2%}
- RSI (14-day): {metrics['rsi']:.1f}
- PE Ratio: {metrics['pe']}
- Market Cap: {metrics['market_cap']}
- Sector: {metrics['sector']}

═══════════════════════════════════════════════════
🏢 COMPANY PROFILE
═══════════════════════════════════════════════════
{company_summary}

═══════════════════════════════════════════════════
💰 DETAILED FINANCIALS
═══════════════════════════════════════════════════
{fin_text if fin_text else 'Limited financial data available.'}

═══════════════════════════════════════════════════
📰 RECENT NEWS HEADLINES
═══════════════════════════════════════════════════
{news_text}

═══════════════════════════════════════════════════
📈 ANALYST RECOMMENDATIONS
═══════════════════════════════════════════════════
{recs_text}

═══════════════════════════════════════════════════
💵 RECENT QUARTERLY EARNINGS
═══════════════════════════════════════════════════
{earnings_text}

═══════════════════════════════════════════════════

Now produce a COMPREHENSIVE investment research report with these sections. Use markdown formatting.
Be direct, data-driven, and extremely specific. Reference actual numbers from the data above.

## 📋 Executive Summary
A 3-4 sentence summary of the overall investment case. Include the verdict: 🟢 BUY / 🟡 HOLD / 🔴 SELL

## 🏢 Company Deep Dive
- What the company does, competitive advantages (moat), market position
- Industry dynamics and where this company fits
- Key business segments and revenue drivers

## 📊 Financial Health Analysis
- Revenue and profitability trends
- Balance sheet strength (debt, cash, current ratio)
- Cash flow analysis
- Valuation assessment (PE, PEG, forward PE vs industry)
- Earnings trajectory

## 📰 News & Sentiment Analysis  
- Summarize the recent news and its potential impact on the stock
- Overall market sentiment: Bullish / Bearish / Neutral
- Any catalysts or red flags from recent developments

## 🔧 Technical Analysis
- RSI interpretation and what it signals now
- Price position relative to 52-week range
- Volatility assessment
- Moving average trend analysis

## ⚠️ Risk Matrix
Rate each risk as HIGH / MEDIUM / LOW:
1. Market/Macro Risk
2. Company-Specific Risk  
3. Regulatory/Political Risk
4. Competition Risk
5. Valuation Risk

## 🎯 Price Target & Recommendation
- Your 6-month and 12-month price targets with reasoning
- Clear BUY / HOLD / SELL recommendation with conviction level (High/Medium/Low)
- Entry price suggestions and stop-loss levels
- Who should buy this stock (risk appetite profile)

## 📌 Key Takeaways
Top 5 bullet points an investor MUST know before making a decision.

Remember: Be quantitative, reference specific data points, and give actionable insights. No generic filler."""

    response = gemini_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=agentic_prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4000,
        )
    )
    return response.text


# ═══════════════════════════════════════════════════════════
# MAIN ANALYSIS LOGIC
# ═══════════════════════════════════════════════════════════

if analyze_btn and company_name:
    # Step 1: Find the ticker
    with st.spinner(f"🔍 Searching for **{company_name}**..."):
        ticker_sym, full_name = get_ticker(company_name)

    if not ticker_sym:
        st.error("❌ Company not found. Try a more specific name like 'Adani Enterprises', 'Tesla Inc', or 'Apple Inc'.")
        st.stop()

    st.success(f"✅ Found: **{full_name}** (`{ticker_sym}`)")

    # Step 2: Fetch market data
    with st.spinner("📊 Fetching live market data..."):
        stock = yf.Ticker(ticker_sym, session=_yf_session)
        df = safe_download(ticker_sym, start=start_date, end=end_date)
        info = safe_get_info(stock)  # Rate-limit safe with retries

    if df.empty:
        st.error("📉 No data found for this date range. Try expanding the date range.")
        st.stop()

    # Handle MultiIndex columns from yf.download
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = safe_squeeze(df["Close"])
    volume = safe_squeeze(df["Volume"])
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    daily_return = close.pct_change()
    volatility = float(np.std(daily_return.dropna()) * np.sqrt(252))
    rsi = calculate_rsi(close)

    # Graceful fallback: if info is empty (rate limited), use historical data
    current_price = info.get('currentPrice') or info.get('regularMarketPrice') or float(close.iloc[-1])
    high_52w = info.get('fiftyTwoWeekHigh') or float(close.max())
    low_52w = info.get('fiftyTwoWeekLow') or float(close.min())
    rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    if not info:
        st.warning("⚠️ Yahoo Finance rate limit hit — using historical data for metrics. Analysis will still proceed.")

    metrics = {
        'current_price': current_price,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'volatility': volatility,
        'rsi': rsi_val,
        'pe': info.get('trailingPE', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        'sector': info.get('sector', 'N/A')
    }

    # ─── Key Metrics Row ─────────────────────────────────
    st.markdown("---")
    st.subheader(f"📊 {full_name} — Key Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Price", f"{current_price:,.2f}")
    c2.metric("📈 52W High", f"{high_52w:,.2f}")
    c3.metric("📉 52W Low", f"{low_52w:,.2f}")
    c4.metric("🌊 Volatility", f"{volatility:.2%}")
    c5.metric("📐 RSI (14)", f"{rsi_val:.1f}")
    c6.metric("🏭 Sector", str(metrics['sector']))

    # RSI Signal
    if rsi_val > 70:
        st.markdown('<div class="signal-sell">⚠️ RSI above 70 — Overbought Signal</div>', unsafe_allow_html=True)
    elif rsi_val < 30:
        st.markdown('<div class="signal-buy">💡 RSI below 30 — Oversold (Potential Buy Zone)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-hold">📊 RSI in neutral range (30-70)</div>', unsafe_allow_html=True)

    # ─── Charts ──────────────────────────────────────────
    st.markdown("---")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Price + Moving Averages
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0a0a1a')
        ax.set_facecolor('#0a0a1a')
        ax.plot(close, label="Close Price", color='#00d4ff', linewidth=1.5)
        ax.plot(ma50, label="50-Day MA", linestyle="--", color='#ffa500', linewidth=1)
        ax.plot(ma200, label="200-Day MA", linestyle="--", color='#ff4444', linewidth=1)
        ax.fill_between(close.index, close.min(), close, alpha=0.05, color='#00d4ff')
        ax.legend(facecolor='#1a1a2e', labelcolor='white', framealpha=0.9)
        ax.set_title(f"{full_name} — Price + Moving Averages", color='white', fontsize=13, fontweight='bold')
        ax.tick_params(colors='#8892b0')
        ax.grid(True, alpha=0.1, color='#667eea')
        for spine in ax.spines.values():
            spine.set_color('#333')
        st.pyplot(fig)
        plt.close()

        # Volume
        fig2, ax2 = plt.subplots(figsize=(12, 3))
        fig2.patch.set_facecolor('#0a0a1a')
        ax2.set_facecolor('#0a0a1a')
        colors = ['#667eea' if daily_return.iloc[i] >= 0 else '#ff4444'
                  for i in range(len(daily_return))]
        ax2.bar(df.index, volume, color=colors, alpha=0.7, width=1.5)
        ax2.set_title("Trading Volume", color='white', fontsize=13, fontweight='bold')
        ax2.tick_params(colors='#8892b0')
        ax2.grid(True, alpha=0.1, color='#667eea')
        for spine in ax2.spines.values():
            spine.set_color('#333')
        st.pyplot(fig2)
        plt.close()

    with col_right:
        # RSI Chart
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor('#0a0a1a')
        ax3.set_facecolor('#0a0a1a')
        ax3.plot(rsi, color='#00ff88', linewidth=1.5)
        ax3.axhline(70, color='#ff4444', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(30, color='#00ff88', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.fill_between(rsi.index, 30, 70, alpha=0.05, color='#667eea')
        ax3.set_ylim(0, 100)
        ax3.set_title("RSI (14-Day)", color='white', fontsize=12, fontweight='bold')
        ax3.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=7)
        ax3.tick_params(colors='#8892b0')
        ax3.grid(True, alpha=0.1, color='#667eea')
        for spine in ax3.spines.values():
            spine.set_color('#333')
        st.pyplot(fig3)
        plt.close()

        # Return Distribution
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        fig4.patch.set_facecolor('#0a0a1a')
        ax4.set_facecolor('#0a0a1a')
        ax4.hist(daily_return.dropna(), bins=50, color='#667eea', alpha=0.8, edgecolor='#764ba2')
        ax4.axvline(0, color='#ff4444', linestyle='--', alpha=0.5)
        ax4.set_title("Daily Return Distribution", color='white', fontsize=12, fontweight='bold')
        ax4.tick_params(colors='#8892b0')
        ax4.grid(True, alpha=0.1, color='#667eea')
        for spine in ax4.spines.values():
            spine.set_color('#333')
        st.pyplot(fig4)
        plt.close()

    # ─── AI Analysis Section ─────────────────────────────
    st.markdown("---")

    if gemini_client:
        # Standard AI Analysis
        st.subheader("🤖 AI Analysis")
        with st.spinner("🧠 Generating AI analysis — Buy / Hold / Sell recommendation..."):
            try:
                analysis = get_ai_analysis(ticker_sym, full_name, metrics)
                st.markdown(analysis)
            except Exception as e:
                st.error(f"AI analysis failed: {e}")

        # Agentic Deep Research
        if enable_agent:
            st.markdown("---")
            st.subheader("🧠 Agentic AI — Deep Investment Research")
            st.markdown("""
            <div class="agent-card">
                <p style="color: #667eea; font-weight: 600; margin-bottom: 10px;">
                    🤖 AI Agent is autonomously researching this company...
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Agent Step 1: Fetch News
            agent_status = st.empty()
            agent_status.markdown('<div class="agent-step">🔎 Step 1/4 — Fetching recent news and headlines...</div>', unsafe_allow_html=True)

            news_headlines = fetch_news_headlines(company_name, ticker_sym)

            # Display fetched news
            if news_headlines:
                with st.expander(f"📰 Recent News ({len(news_headlines)} headlines found)", expanded=False):
                    for h in news_headlines:
                        st.markdown(f"• **{h['title']}** — _{h['publisher']}_")

            # Agent Step 2: Deep Financials
            agent_status.markdown('<div class="agent-step">📊 Step 2/4 — Gathering detailed financial data...</div>', unsafe_allow_html=True)

            financial_details = get_financial_details(ticker_sym)

            # Display key financials
            fin = financial_details.get('financials', {})
            if fin:
                with st.expander("💰 Detailed Financial Data", expanded=False):
                    fin_col1, fin_col2, fin_col3 = st.columns(3)
                    with fin_col1:
                        st.markdown("**Profitability**")
                        for key in ['gross_margins', 'operating_margins', 'profit_margins', 'return_on_equity']:
                            val = fin.get(key, 'N/A')
                            if val != 'N/A' and val is not None:
                                st.write(f"• {key.replace('_', ' ').title()}: {val:.2%}" if isinstance(val, float) and abs(val) < 10 else f"• {key.replace('_', ' ').title()}: {val}")
                    with fin_col2:
                        st.markdown("**Balance Sheet**")
                        for key in ['debt_to_equity', 'current_ratio', 'total_cash', 'total_debt']:
                            val = fin.get(key, 'N/A')
                            if val != 'N/A' and val is not None:
                                if isinstance(val, (int, float)) and abs(val) >= 1e9:
                                    st.write(f"• {key.replace('_', ' ').title()}: ${val/1e9:.2f}B")
                                elif isinstance(val, (int, float)) and abs(val) >= 1e6:
                                    st.write(f"• {key.replace('_', ' ').title()}: ${val/1e6:.2f}M")
                                else:
                                    st.write(f"• {key.replace('_', ' ').title()}: {val}")
                    with fin_col3:
                        st.markdown("**Growth & Valuation**")
                        for key in ['revenue_growth', 'earnings_growth', 'forward_pe', 'peg_ratio']:
                            val = fin.get(key, 'N/A')
                            if val != 'N/A' and val is not None:
                                if isinstance(val, float) and abs(val) < 10:
                                    st.write(f"• {key.replace('_', ' ').title()}: {val:.2f}")
                                else:
                                    st.write(f"• {key.replace('_', ' ').title()}: {val}")

            # Agent Step 3: Analyst Data
            agent_status.markdown('<div class="agent-step">🔬 Step 3/4 — Analyzing analyst recommendations & earnings...</div>', unsafe_allow_html=True)

            # Agent Step 4: Generate Deep Report
            agent_status.markdown('<div class="agent-step">🧠 Step 4/4 — AI generating comprehensive investment report...</div>', unsafe_allow_html=True)

            with st.spinner("🧠 Generating deep investment report... (this may take 15-30 seconds)"):
                try:
                    deep_report = run_agentic_analysis(
                        ticker_sym, full_name, metrics,
                        news_headlines, financial_details
                    )
                    agent_status.markdown(
                        '<div class="agent-step" style="border-left-color: #00ff88;">✅ Agent completed — Deep report generated successfully!</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(deep_report)
                except Exception as e:
                    agent_status.markdown(
                        f'<div class="agent-step" style="border-left-color: #ff4444;">❌ Agent error: {e}</div>',
                        unsafe_allow_html=True
                    )
                    st.error(f"Agentic analysis failed: {e}")
    else:
        st.subheader("🤖 AI Analysis")
        st.warning("⚠️ AI engine not configured. The app owner needs to set the GEMINI_API_KEY in the .env file.")

    # ─── Raw Data Table ──────────────────────────────────
    st.markdown("---")
    with st.expander("📋 Raw Data (Last 20 Trading Days)", expanded=False):
        raw_df = pd.DataFrame({
            "Close": close,
            "MA50": ma50,
            "MA200": ma200,
            "RSI": rsi,
            "Daily Return %": daily_return * 100
        }).tail(20).round(2)
        st.dataframe(raw_df, use_container_width=True)

elif analyze_btn and not company_name:
    st.warning("⚠️ Please enter a company name in the sidebar to begin analysis.")