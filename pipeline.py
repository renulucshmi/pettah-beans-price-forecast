"""Shared code for the notebook (Step 11) and the Streamlit app.

Rebuilds the model's input features from the weekly data exactly as in Step 3,
and turns the saved RF + ARIMA hybrid into a forecast for next week.
"""
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "chg_1", "chg_2", "chg_3", "chg_4", "gap_ma4", "vol_4",
    "dambulla_gap", "dambulla_chg", "carrot_chg", "cabbage_chg",
    "rain_ne_0", "rain_ne_1", "rain_ne_2", "rain_bd_0", "rain_bd_1", "rain_bd_2",
    "month_sin", "month_cos", "avurudu", "december",
]


def make_features(df):
    """Same features as Step 3 of the notebook, for every week in df (no rows dropped)."""
    logp = np.log(df["beans_pettah"])
    feat = pd.DataFrame(index=df.index)
    feat["target"] = logp.shift(-1) - logp

    feat["chg_1"] = logp.diff()
    feat["chg_2"] = feat["chg_1"].shift(1)
    feat["chg_3"] = feat["chg_1"].shift(2)
    feat["chg_4"] = feat["chg_1"].shift(3)
    feat["gap_ma4"] = logp - logp.rolling(4).mean()
    feat["vol_4"] = feat["chg_1"].rolling(4).std()

    feat["dambulla_gap"] = np.log(df["beans_dambulla"]) - logp
    feat["dambulla_chg"] = np.log(df["beans_dambulla"]).diff()
    feat["carrot_chg"] = np.log(df["carrot_pettah"]).diff()
    feat["cabbage_chg"] = np.log(df["cabbage_pettah"]).diff()

    for name, col in [("ne", "rain_nuwaraeliya"), ("bd", "rain_badulla")]:
        feat[f"rain_{name}_0"] = df[col]
        feat[f"rain_{name}_1"] = df[col].shift(1)
        feat[f"rain_{name}_2"] = df[col].shift(2)

    next_week = df.index + pd.Timedelta(days=7)
    feat["month_sin"] = np.sin(2 * np.pi * next_week.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * next_week.month / 12)
    feat["avurudu"] = (next_week.month == 4).astype(int)
    feat["december"] = (next_week.month == 12).astype(int)

    feat["price_now"] = df["beans_pettah"]
    feat["price_next"] = df["beans_pettah"].shift(-1)
    return feat


def forecast_next_week(bundle, weekly):
    """Forecast the week after the last row of `weekly` with the saved hybrid.

    Returns a dict with the ARIMA part, the forest's correction and the final forecast.
    """
    log_price = np.log(weekly["beans_pettah"])
    log_price.index.freq = "W-FRI"

    x = make_features(weekly)[bundle["feature_cols"]].iloc[[-1]]
    if x.isna().any(axis=None):
        raise ValueError("Not enough weeks of data to build the features (need at least 5).")

    # ARIMA: reuse the saved parameters on this price history, forecast one week ahead
    arima_next = bundle["arima"].apply(log_price).forecast(1).iloc[0]
    arima_change = arima_next - log_price.iloc[-1]
    correction = bundle["rf"].predict(x)[0]
    change = arima_change + correction

    price_now = weekly["beans_pettah"].iloc[-1]
    return {
        "week": weekly.index[-1] + pd.Timedelta(days=7),
        "price_now": price_now,
        "arima_price": price_now * np.exp(arima_change),
        "arima_change": arima_change,
        "rf_correction": correction,
        "change": change,
        "price_next": price_now * np.exp(change),
    }
