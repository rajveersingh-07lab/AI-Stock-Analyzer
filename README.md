# 📈 AI Stock Analyzer

An AI-powered stock analysis tool that lets you type any company name and get instant, comprehensive investment analysis — powered by Google Gemini AI.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit)
![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-4285F4?logo=google)

## ✨ Features

- 🔍 **Smart Ticker Search** — Type any company name (e.g. "Adani", "Tesla") and it auto-finds the stock ticker
- 📊 **Live Market Data** — Fetches real-time data via yfinance
- 📈 **Interactive Charts** — Price + Moving Averages (50/200), Volume, RSI, Return Distribution
- 📋 **Key Metrics** — Current Price, 52W High/Low, Volatility, RSI, Sector
- 🤖 **AI Analysis** — Buy/Hold/Sell recommendation powered by Google Gemini
- 🧠 **Agentic Deep Research** — AI autonomously researches news, financials, and generates a full investment report
- 🌑 **Dark Theme UI** — Beautiful, modern dark-themed interface

## 🚀 Live Demo

👉 [**Try it here**](YOUR_STREAMLIT_CLOUD_URL)

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data**: yfinance, Pandas, NumPy
- **Charts**: Matplotlib
- **AI Engine**: Google Gemini (gemini-2.5-flash)
- **News**: yfinance News + Google News RSS

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/AI-Stock-Analyzer.git
cd AI-Stock-Analyzer
pip install -r requirements.txt
```

Create a `.env` file and add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

Get a free API key from [Google AI Studio](https://aistudio.google.com)

## ▶️ Run Locally

```bash
streamlit run APP.py
```

## 📸 Screenshots

<!-- Add screenshots of your app here -->

## 👨‍💻 Author

**Rajveer** — BBA Finance Student | AI & Finance Enthusiast

## ⚠️ Disclaimer

This tool is for educational and informational purposes only. It is not financial advice. Always do your own research before making investment decisions.
