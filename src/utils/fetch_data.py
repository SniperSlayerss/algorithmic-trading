import pandas as pd
import requests
import time

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", days=100):
    base_url = "https://api.binance.com/api/v3/klines"
    ms_per_day = 24 * 60 * 60 * 1000
    end_time = int(time.time() * 1000)
    start_time = end_time - days * ms_per_day
    limit = 1000  # Binance max limit per request

    all_data = []

    while start_time < end_time:
        print(len(all_data))
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": limit
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print("Error fetching data:", response.text)
            break

        data = response.json()
        if not data:
            break

        all_data.extend(data)
        start_time = data[-1][0] + 1  # Move start time forward

    return pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "trades",
        "taker_base_volume", "taker_quote_volume", "ignore"
    ])

# Fetch 1 year of 15-minute data
df = fetch_binance_klines()

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

# Save to CSV
df.to_csv("btc_usd_1m.csv", index=False)
print("Data saved to btc_usd_1m.csv")
