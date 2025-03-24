from lstm_strategy import LSTMStrategy
from pathlib import Path
import numpy as np

file_path = Path(__file__)
data_path = file_path.parents[1] / "data" / "btc_usd_1m_24_03_2025.csv"
financial_data = LSTMStrategy.fetch_data(data_path)
financial_data["ema_short"] = financial_data["close"].ewm(span=20, adjust=False).mean()
financial_data["ema_long"] = financial_data["close"].ewm(span=100, adjust=False).mean()
# LSTMStrategy.save_and_train_model(financial_data)

# lstm_model = LSTMStrategy()
#
# if lstm_model is not None:
#     print(lstm_model.is_entry(financial_data))

LSTMStrategy.run_strategy(financial_data=financial_data)
