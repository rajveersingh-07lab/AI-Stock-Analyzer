import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
ticker = "RELIANCE.NS"
df = yf.download(ticker, start="2023-01-01", end="2024-12-31")

df["MA50"] = df["Close"].rolling(50).mean()
df["MA200"] = df["Close"].rolling(200).mean()

df["Daily Return"] = df["Close"].pct_change()
volatility = df["Daily Return"].std() * (252**0.5)

plt.figure(figsize=(12,6))
plt.plot(df["Close"].squeeze(), label="Close Price")
plt.plot(df["MA50"].squeeze(), label="50-Day MA", linestyle="--")
plt.plot(df["MA200"].squeeze(), label="200-Day MA", linestyle="--")
plt.title(f"{ticker} | Volatility: {volatility.squeeze():.2%}")
plt.legend()
plt.savefig("chart.png")
plt.show()