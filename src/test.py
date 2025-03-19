from lstm_strategy import LSTMStrategy
from pathlib import Path
import numpy as np

file_path = Path(__file__)
data_path = file_path.parents[1] / "data" / "btc_usd_1m.csv"
financial_data = LSTMStrategy.fetch_data(data_path)
financial_data["ema_short"] = financial_data["close"].ewm(span=20, adjust=False).mean()
financial_data["ema_long"] = financial_data["close"].ewm(span=100, adjust=False).mean()
# LSTMStrategy.save_and_train_model(financial_data)

model = LSTMStrategy.load_model()

test_input = np.array(financial_data[["close", "ema_short", "ema_long"]].iloc[-51:-1])
test_input = test_input.reshape(1, 50, 3)
print(model.predict(test_input))
