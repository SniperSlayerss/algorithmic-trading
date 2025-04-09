from typing import Optional, Tuple
import math
import os
import sys
import pickle

import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from keras._tf_keras.keras import models, layers, optimizers, Input
from sklearn.preprocessing import MinMaxScaler

from strategy.strategy import Strategy

MODEL_SAVE_PATH = "data/lstm_model.keras"
SCALER_SAVE_PATH = "data/scaler.pkl"
SEQ_LEN = 50


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(data.shape[0] - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


"""
Notes:
- https://www.investopedia.com/ask/answers/122314/how-do-i-use-exponential-moving-average-ema-create-forex-trading-strategy.asp
  > Short-term EMA crosses over long-term -> bullish signal

Possibly split single strategys into multiple classes,
then can run each strategy on their own or combine them..!

Focus:
- Focus on each single strategy, for example LSTM net, or EMA crossover
- Then create strategys combining these, like an ensemble of strategys
"""


class LSTMStrategy(Strategy):
    model: Optional[models.Model] = None
    scaler: Optional[MinMaxScaler] = None

    @classmethod
    def run_strategy(cls, financial_data: pd.DataFrame) -> np.ndarray:
        lstm_strategy = cls()
        entries = []
        i = 0
        while i < len(financial_data) - SEQ_LEN:
            split_data = financial_data.iloc[: SEQ_LEN + i + 1]
            if lstm_strategy.is_entry(split_data):
                entries.append(split_data.iloc[-1])
            i += 1
        return np.array(entries)

    def is_entry(self, financial_data: pd.DataFrame) -> bool:
        if len(financial_data) < SEQ_LEN:
            print(
                f"ERROR: len of {len(financial_data)} is less than sequence length of {SEQ_LEN}"
            )
            return False

        features = financial_data[["close", "ema_short", "ema_long"]].iloc[-51:-1]
        pct_change = features.pct_change().fillna(0)

        if self.scaler is None:
            self.scaler = LSTMStrategy.load_scaler()

        if self.model is None:
            self.model = LSTMStrategy.load_model()

        x = np.array(pct_change)
        # x = self.scaler.fit_transform(x)
        x = x.reshape(1, SEQ_LEN, 3)

        y = self.model.predict(x, verbose=0)

        """
        Check if price is increaing
        Check if crossover is bullish
        [close, ema_short, ema_long]
        """
        return x[0, -1, 0] < y[0, 0]

    @staticmethod
    def load_model() -> models.Model:
        try:
            model = models.load_model(MODEL_SAVE_PATH)
            print(f"INFO: Loaded '{MODEL_SAVE_PATH}'")
        except ValueError:
            print(f"ERROR: File not found '{MODEL_SAVE_PATH}'")
            print("WARNING: Train and save the model before attempting to load")
            sys.exit(1)
        return model

    @staticmethod
    def load_scaler() -> MinMaxScaler:
        try:
            with open(SCALER_SAVE_PATH, "rb") as f:
                scaler = pickle.load(f)
            print(f"INFO: Loaded '{SCALER_SAVE_PATH}'")
        except FileNotFoundError:
            print(f"ERROR: File not found '{SCALER_SAVE_PATH}'")
            print("WARNING: Train and save the network before attempting to load")
            sys.exit(1)
        return scaler

    @staticmethod
    def save_and_train_model(financial_data: pd.DataFrame) -> None:
        model, scaler = LSTMStrategy.train_model(financial_data)

        model.save(MODEL_SAVE_PATH)
        print(f"INFO: Saved model '{MODEL_SAVE_PATH}'")

        with open(SCALER_SAVE_PATH, "wb") as f:
            pickle.dump(scaler, f)
        print(f"INFO: Saved scaler '{SCALER_SAVE_PATH}'")

    @staticmethod
    def train_model(financial_data: pd.DataFrame) -> Tuple[models.Model, MinMaxScaler]:
        # ~~~ INDICATORS ~~~
        financial_data["ema_short"] = (
            financial_data["close"].ewm(span=20, adjust=False).mean()
        )
        financial_data["ema_long"] = (
            financial_data["close"].ewm(span=100, adjust=False).mean()
        )

        # TODO: improve data split
        features = financial_data[["close", "ema_short", "ema_long"]]
        financial_data_pct = np.array(features.pct_change().fillna(0))

        # ~~~ PREPROCESSING ~~~
        scaler = MinMaxScaler()
        # train_data = scaler.fit_transform(train_data)

        train_size = int(0.8 * len(financial_data_pct))

        train_data = financial_data_pct[:train_size]
        x_train, y_train = create_sequences(train_data, SEQ_LEN)

        test_data = financial_data_pct[train_size - SEQ_LEN :]
        x_test, y_test = create_sequences(test_data, SEQ_LEN)

        normalizer = layers.Normalization(axis=-1)
        normalizer.adapt(x_train)

        # ~~~ MODEL ~~~
        model = models.Sequential(
            [
                Input(shape=(x_train.shape[1], x_train.shape[2])),
                normalizer,
                layers.LSTM(units=128, return_sequences=True),
                layers.LSTM(units=64, return_sequences=True),
                layers.LSTM(units=64, return_sequences=False),
                layers.Dense(units=3),
            ]
        )

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.01), loss="mean_squared_error"
        )

        # ~~~ TRAINING ~~~
        model.fit(
            x_train,
            y_train,
            epochs=4,
            batch_size=32,
            validation_data=(x_test, y_test),
        )

        # ~~~ EVALUATION ~~~
        trainScore = model.evaluate(x_train, y_train, verbose=0)
        print(
            "INFO: Train Score: %.2f MSE (%.2f RMSE)"
            % (trainScore, math.sqrt(trainScore))
        )
        testScore = model.evaluate(x_test, y_test, verbose=0)
        print(
            "INFO: Test Score: %.2f MSE (%.2f RMSE)" % (testScore, math.sqrt(testScore))
        )

        return (model, scaler)
