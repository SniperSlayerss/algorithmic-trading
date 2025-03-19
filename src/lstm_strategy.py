from pathlib import Path
from typing import Optional
import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd

import tensorflow as tf
from keras._tf_keras.keras import models, layers, optimizers, Input
from sklearn.preprocessing import MinMaxScaler

from strategy import Strategy

MODEL_SAVE_PATH = "lstm_model.keras"


def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(data.shape[0] - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


class LSTMStrategy(Strategy):
    def run_strategy(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        pass

    def is_entry(self, financial_data: pd.DataFrame) -> bool:
        pass

    @staticmethod
    def load_model() -> Optional[models.Model]:
        model = None
        try:
            model = models.load_model(MODEL_SAVE_PATH)
            print(f"INFO: Loaded '{MODEL_SAVE_PATH}'")
        except ValueError:
            print(f"ERROR: File not found '{MODEL_SAVE_PATH}'")
            print("WARNING: Train and save the model before attempting to load")

        return model

    @staticmethod
    def save_and_train_model(financial_data: Path) -> None:
        model = LSTMStrategy.train_model(financial_data)
        model.save(MODEL_SAVE_PATH)
        print(f"INFO: Saved model '{MODEL_SAVE_PATH}'")

    @staticmethod
    def train_model(financial_data: Path) -> models.Model:
        # ~~~ INDICATORS ~~~
        financial_data["ema_short"] = (
            financial_data["close"].ewm(span=20, adjust=False).mean()
        )
        financial_data["ema_long"] = (
            financial_data["close"].ewm(span=100, adjust=False).mean()
        )

        # TODO: improve data split
        train_data = financial_data[["close", "ema_short", "ema_long"]].iloc[0:2000]
        test_data = financial_data[["close", "ema_short", "ema_long"]].iloc[2000:3000]

        # ~~~ PREPROCESSING ~~~
        scaler = MinMaxScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        x_train, y_train = create_sequences(train_data, 50)
        x_test, y_test = create_sequences(train_data, 50)

        # ~~~ MODEL ~~~
        model = models.Sequential()
        model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))
        model.add(layers.LSTM(units=128, return_sequences=True))
        model.add(layers.LSTM(units=64, return_sequences=True))
        model.add(layers.LSTM(units=64, return_sequences=False))
        model.add(layers.Dense(units=3))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.01), loss="mean_squared_error"
        )

        # ~~~ TRAINING ~~~
        model.fit(
            x_train,
            y_train,
            epochs=10,
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

        return model
