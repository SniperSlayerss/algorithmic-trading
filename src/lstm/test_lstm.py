from pathlib import Path

import numpy as np
from lstm.lstm_strategy import LSTMStrategy

from bokeh.plotting import figure, show

file_path = Path(__file__)
data_path = file_path.parents[2] / "data" / "btc_usd_1m_24_03_2025.csv"
financial_data = LSTMStrategy.fetch_data(data_path)
financial_data["ema_short"] = financial_data["close"].ewm(span=20, adjust=False).mean()
financial_data["ema_long"] = financial_data["close"].ewm(span=100, adjust=False).mean()

# LSTMStrategy.save_and_train_model(financial_data.iloc[:1000])
entry_points = LSTMStrategy.run_strategy(financial_data=financial_data.iloc[1000:1150])

subset_of_financial_data = financial_data.iloc[1000:3000][["timestamp", "close"]]
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
p = figure(x_axis_type="datetime", tools=TOOLS, width=2000, height=1000)
p.line(
    x=subset_of_financial_data["timestamp"],
    y=subset_of_financial_data["close"],
    color="gray",
    legend_label="price",
)

p.title.text = "BTC price close over time"
p.xgrid.grid_line_color = None
p.xaxis.axis_label = "Time"
p.yaxis.axis_label = "Close"
p.legend.level = "overlay"
p.legend.location = "top_left"

show(p)
