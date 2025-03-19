from pathlib import Path

import pandas as pd
import plotly.graph_objs as go

file_path = Path(__file__)
data_path = file_path.parents[2] / "data" / "btc_usd_1m.csv"

df = pd.read_csv(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

ema_span_short = 20
ema_span_long = 100
df["ema_short"] = df["close"].ewm(span=ema_span_short, adjust=False).mean()
df["ema_long"] = df["close"].ewm(span=ema_span_long, adjust=False).mean()

df_s = df.iloc[-2000:-1]

fig = go.Figure(
    data=[
        go.Candlestick(
            x=df_s["timestamp"],
            open=df_s["open"],
            high=df_s["high"],
            low=df_s["low"],
            close=df_s["close"],
        )
    ]
)

fig.add_trace(
    go.Scatter(
        x=df_s["timestamp"],
        y=df_s["ema_short"],
        mode="lines",
        line=dict(width=2),
        name=f"EMA (span={ema_span_short})",
    )
)

fig.add_trace(
    go.Scatter(
        x=df_s["timestamp"],
        y=df_s["ema_long"],
        mode="lines",
        line=dict(width=2),
        name=f"EMA (span={ema_span_long})",
    )
)

fig.update_layout(
    title="Candlestick Chart with EMA Indicator",
    xaxis_title="Timestamp",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
)

fig.show()
