"""Streamlit dashboard for the Pettah beans price forecast.

Run from the project folder:   streamlit run app.py
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pipeline import forecast_next_week

BASE = Path(__file__).parent
st.set_page_config(page_title="Pettah Beans Price Forecast", layout="wide")

# Colours: one fixed colour per model family (same order as the notebook's charts)
FAMILY_COLORS = {
    "Baseline": "#2a78d6", "Single model": "#eb6834", "Bagging": "#1baf7a",
    "Boosting": "#eda100", "Stacking / voting": "#e87ba4", "Hybrid": "#008300",
}
PETTAH, DAMBULLA, FORECAST = "#2a78d6", "#eb6834", "#008300"

FEATURE_NAMES = {
    "chg_1": "Price change this week", "chg_2": "Price change last week",
    "chg_3": "Price change 2 weeks ago", "chg_4": "Price change 3 weeks ago",
    "gap_ma4": "Price vs its 4-week average", "vol_4": "Recent volatility (4 weeks)",
    "dambulla_gap": "Dambulla vs Pettah price gap", "dambulla_chg": "Dambulla price change this week",
    "carrot_chg": "Carrot price change this week", "cabbage_chg": "Cabbage price change this week",
    "rain_ne_0": "Nuwara Eliya rain this week", "rain_ne_1": "Nuwara Eliya rain last week",
    "rain_ne_2": "Nuwara Eliya rain 2 weeks ago", "rain_bd_0": "Badulla rain this week",
    "rain_bd_1": "Badulla rain last week", "rain_bd_2": "Badulla rain 2 weeks ago",
    "month_sin": "Season (month, sine)", "month_cos": "Season (month, cosine)",
    "avurudu": "April (Sinhala and Tamil New Year)", "december": "December",
}


@st.cache_data
def load_weekly():
    return pd.read_csv(BASE / "dataset" / "weekly_with_rain.csv", index_col="date", parse_dates=True)


@st.cache_data
def load_results():
    return pd.read_csv(BASE / "results" / "results_final.csv", index_col="model")


@st.cache_data
def load_test_predictions():
    return pd.read_csv(BASE / "results" / "test_predictions.csv", index_col="week", parse_dates=True)


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model" / "hybrid_rf_arima.joblib")


weekly = load_weekly()
results = load_results()
test_pred = load_test_predictions()
bundle = load_model()
scores = bundle["test_scores"]

st.title("Pettah Beans Price Forecast")
st.caption("Next week's average wholesale price of beans at Pettah market, Colombo, forecast with an "
           "ensemble of a Random Forest and ARIMA. Prakasan R. (224152U), University of Moratuwa.")

tab_data, tab_models, tab_features, tab_forecast = st.tabs(
    ["Data overview", "Model comparison", "What drives the price", "Forecast"])

# ---------------------------------------------------------------- data overview
with tab_data:
    last, prev = weekly.iloc[-1], weekly.iloc[-2]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Pettah beans, week ending {weekly.index[-1]:%d %b %Y}", f"Rs {last.beans_pettah:,.0f}/kg",
              f"{last.beans_pettah - prev.beans_pettah:+,.0f} vs previous week", delta_color="inverse")
    c2.metric("Dambulla beans, same week", f"Rs {last.beans_dambulla:,.0f}/kg")
    c3.metric(f"Weeks of data ({weekly.index[0]:%b %Y} to {weekly.index[-1]:%b %Y})", f"{len(weekly)}")
    c4.metric("Weeks filled by interpolation",
              f"{int(weekly.was_missing.sum())} ({weekly.was_missing.mean():.1%})")

    period = st.radio("Period", ["Last year", "Last 3 years", "All"], index=1, horizontal=True)
    years = {"Last year": 1, "Last 3 years": 3, "All": 20}[period]
    shown = weekly[weekly.index > weekly.index[-1] - pd.DateOffset(years=years)]

    prices = (shown[["beans_pettah", "beans_dambulla"]]
              .rename(columns={"beans_pettah": "Pettah", "beans_dambulla": "Dambulla"})
              .reset_index().melt("date", var_name="Market", value_name="Rs/kg"))
    st.subheader("Weekly wholesale price of beans")
    st.altair_chart(
        alt.Chart(prices).mark_line(strokeWidth=2).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
            y=alt.Y("Rs/kg:Q", title="Rs per kg"),
            color=alt.Color("Market:N", scale=alt.Scale(domain=["Pettah", "Dambulla"], range=[PETTAH, DAMBULLA]),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("date:T", title="Week ending"), "Market:N", alt.Tooltip("Rs/kg:Q", format=",.0f")],
        ).properties(height=320),
        width="stretch")

    rain = (shown[["rain_nuwaraeliya", "rain_badulla"]]
            .rename(columns={"rain_nuwaraeliya": "Nuwara Eliya", "rain_badulla": "Badulla"})
            .reset_index().melt("date", var_name="Place", value_name="mm"))
    st.subheader("Weekly rainfall in the main growing areas")
    st.altair_chart(
        alt.Chart(rain).mark_line(strokeWidth=1.5).encode(
            x=alt.X("date:T", title=None, axis=alt.Axis(format="%b %Y")),
            y=alt.Y("mm:Q", title="Rain per week (mm)"),
            color=alt.Color("Place:N", scale=alt.Scale(domain=["Nuwara Eliya", "Badulla"], range=[PETTAH, DAMBULLA]),
                            legend=alt.Legend(orient="top")),
            tooltip=[alt.Tooltip("date:T", title="Week ending"), "Place:N", alt.Tooltip("mm:Q", format=",.0f")],
        ).properties(height=240),
        width="stretch")

    with st.expander("About the data"):
        st.markdown(
            "- **Prices:** Central Bank of Sri Lanka, *Daily Price Report*, downloaded from Macro Lab "
            "(macrolab.lk). Daily wholesale prices averaged into weeks ending on Friday.\n"
            "- **Rainfall:** Open-Meteo historical weather archive (ERA5), daily totals for Nuwara Eliya "
            "and Badulla, summed per week.\n"
            "- **Missing weeks** (no trading days reported, e.g. long holidays) were filled by time "
            "interpolation and flagged in `was_missing`.")
        st.dataframe(weekly.tail(10).iloc[::-1].round(1), width="stretch")

# ---------------------------------------------------------------- model comparison
with tab_models:
    best = results.index[0]
    naive = results.loc["Naive (no change)"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final model", best.replace("Hybrid: ", ""))
    c2.metric("Test RMSE (2025-26)", f"Rs {scores['test_rmse']:.1f}/kg",
              f"{scores['test_rmse'] - naive.test_rmse:+.1f} vs naive guess", delta_color="inverse")
    c3.metric("Test MAE", f"Rs {scores['test_mae']:.1f}/kg",
              f"{scores['test_mae'] - naive.test_mae:+.1f} vs naive guess", delta_color="inverse")
    c4.metric("Test R² (weekly change)", f"{scores['test_r2']:.2f}")

    st.markdown(
        "All models were trained on 2016-2023, tuned on 2024 (validation) and tested on 2025-26. "
        "**Unseen RMSE** combines the validation and test weeks (149 weeks the models never trained on) "
        "and is used to rank the models. RMSE and MAE are in Rs per kg; R² is for the weekly % change.")

    families = st.multiselect("Model families", list(FAMILY_COLORS), default=list(FAMILY_COLORS))
    table = results[results.family.isin(families)].rename_axis("Model")

    st.dataframe(
        table[["family", "unseen_rmse", "test_rmse", "test_mae", "test_r2", "val_rmse", "train_rmse",
               "overfit_gap", "train_time_s"]],
        width="stretch",
        height=35 * (len(table) + 1) + 3,      # show every row without scrolling
        column_config={
            "family": "Family",
            "unseen_rmse": st.column_config.NumberColumn("Unseen RMSE", format="%.1f"),
            "test_rmse": st.column_config.NumberColumn("Test RMSE", format="%.1f"),
            "test_mae": st.column_config.NumberColumn("Test MAE", format="%.1f"),
            "test_r2": st.column_config.NumberColumn("Test R²", format="%.2f"),
            "val_rmse": st.column_config.NumberColumn("Validation RMSE", format="%.1f"),
            "train_rmse": st.column_config.NumberColumn("Train RMSE", format="%.1f"),
            "overfit_gap": st.column_config.NumberColumn("Overfit gap", format="%.1f",
                                                         help="Test RMSE minus train RMSE"),
            "train_time_s": st.column_config.NumberColumn("Train time (s)", format="%.2f"),
        })

    chart_kind = st.radio("Chart", ["Test error", "Training vs test error", "Training time"], horizontal=True)
    data = table.reset_index().rename(columns={"Model": "model"})
    order = data.sort_values("test_rmse")["model"].tolist()
    color = alt.Color("family:N", title="Family",
                      scale=alt.Scale(domain=list(FAMILY_COLORS), range=list(FAMILY_COLORS.values())),
                      legend=alt.Legend(orient="top"))
    y = alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False))
    height = max(200, 32 * len(data))

    if chart_kind == "Test error":
        bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3).encode(
            x=alt.X("test_rmse:Q", title="Test RMSE, 2025-26 (Rs/kg), lower is better"), y=y, color=color,
            tooltip=["model", "family", alt.Tooltip("test_rmse:Q", format=".1f", title="Test RMSE"),
                     alt.Tooltip("test_mae:Q", format=".1f", title="Test MAE")])
        rule = alt.Chart(pd.DataFrame({"x": [naive.test_rmse]})).mark_rule(
            strokeDash=[4, 4], color="#888").encode(x="x:Q")
        st.altair_chart((bars + rule).properties(height=height), width="stretch")
        st.caption(f"Dashed line: the naive 'no change' guess (Rs {naive.test_rmse:.1f}).")
    elif chart_kind == "Training vs test error":
        long = data.melt(["model", "family"], ["train_rmse", "test_rmse"], var_name="part", value_name="rmse")
        long["part"] = long["part"].map({"train_rmse": "Training weeks (2016-23)", "test_rmse": "Test weeks (2025-26)"})
        line = alt.Chart(long).mark_line(color="#cccccc", strokeWidth=2).encode(
            x="rmse:Q", y=y, detail="model:N")
        dots = alt.Chart(long).mark_circle(size=80, opacity=1).encode(
            x=alt.X("rmse:Q", title="RMSE (Rs/kg)"), y=y,
            color=alt.Color("part:N", title=None, scale=alt.Scale(range=[PETTAH, DAMBULLA]),
                            legend=alt.Legend(orient="top")),
            tooltip=["model", "part", alt.Tooltip("rmse:Q", format=".1f")])
        st.altair_chart((line + dots).properties(height=height), width="stretch")
        st.caption("A long line means the model fits the training weeks much better than new weeks (overfitting).")
    else:
        timed = data[data.train_time_s > 0]
        st.altair_chart(
            alt.Chart(timed).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("train_time_s:Q", title="Training time in seconds (log scale)", scale=alt.Scale(type="log")),
                y=y, color=color,
                tooltip=["model", alt.Tooltip("train_time_s:Q", format=".2f", title="seconds")],
            ).properties(height=max(200, 32 * len(timed))),
            width="stretch")
        st.caption("Times vary between runs; the naive guess needs no training.")

    st.subheader("Actual vs predicted price on the test weeks (2025-26)")
    pick = st.multiselect("Models", [c for c in test_pred.columns if c != "actual"],
                          default=["Hybrid: RF + ARIMA", "Naive (no change)"])
    shown_pred = test_pred[["actual"] + pick].rename(columns={"actual": "Actual price"})
    long_pred = shown_pred.reset_index().melt("week", var_name="Series", value_name="Rs/kg")
    series = ["Actual price"] + pick
    palette = ["#222222", FORECAST, DAMBULLA, PETTAH, "#eda100", "#e87ba4", "#1baf7a"]
    st.altair_chart(
        alt.Chart(long_pred).mark_line(strokeWidth=2).encode(
            x=alt.X("week:T", title=None, axis=alt.Axis(format="%b %Y")),
            y=alt.Y("Rs/kg:Q", title="Rs per kg"),
            color=alt.Color("Series:N", sort=series,
                            scale=alt.Scale(domain=series, range=palette[:len(series)]),
                            legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.condition(alt.datum.Series == "Actual price", alt.value([1, 0]), alt.value([5, 3])),
            tooltip=[alt.Tooltip("week:T", title="Week ending"), "Series:N", alt.Tooltip("Rs/kg:Q", format=",.0f")],
        ).properties(height=340),
        width="stretch")
    st.caption("Each forecast was made one week earlier, using only data up to that week. "
               "The naive guess is simply last week's price, so it is always one week late.")

    st.markdown(
        "**Key findings**\n"
        "- Every ensemble beats the naive guess; the single decision tree does not (it memorises the training weeks).\n"
        "- Random Forest (bagging) and XGBoost (boosting) reach almost the same accuracy, within about Rs 1.\n"
        "- Voting beat stacking: the tree models make very similar predictions, so learned weights mostly fit noise.\n"
        "- The RF + ARIMA hybrid is best: ARIMA uses only past prices, and the forest adds rain and the other "
        "market, which ARIMA cannot see.")

# ---------------------------------------------------------------- feature importance
with tab_features:
    st.markdown(
        "The final model forecasts next week's price in two parts: **ARIMA** uses only the price history, and "
        "a **Random Forest** corrects ARIMA's forecast using the other information. The chart shows how much the "
        "forest relies on each feature (impurity importance, all weeks).")
    top_n = st.slider("Number of features", 5, len(bundle["feature_cols"]), 10)
    imp = (pd.DataFrame({"feature": bundle["feature_cols"], "importance": bundle["rf"].feature_importances_})
           .assign(name=lambda d: d.feature.map(FEATURE_NAMES))
           .sort_values("importance", ascending=False).head(top_n))
    st.altair_chart(
        alt.Chart(imp).mark_bar(cornerRadiusEnd=3, color=PETTAH).encode(
            x=alt.X("importance:Q", title="Share of the forest's error reduction"),
            y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)),
            tooltip=["name", "feature", alt.Tooltip("importance:Q", format=".3f")],
        ).properties(height=max(200, 28 * top_n)),
        width="stretch")
    st.markdown(
        "- **Rain in Nuwara Eliya this week** is by far the most important input. Heavy rain in the main "
        "growing area damages crops and slows harvesting and transport, so less reaches Pettah the next week.\n"
        "- The **Dambulla market** (its price change and its gap to Pettah) and **recent price movements** "
        "(volatility, this week's change) come next: they help tell whether a jump is likely to continue or be "
        "partly reversed.\n"
        "- **Calendar features** (month, April, December) add little on their own.")

# ---------------------------------------------------------------- forecast
with tab_forecast:
    base = forecast_next_week(bundle, weekly)
    rmse = scores["test_rmse"]

    st.subheader(f"Forecast for the week ending {base['week']:%d %B %Y}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hybrid forecast", f"Rs {base['price_next']:,.0f}/kg",
              f"{base['price_next'] - base['price_now']:+,.0f} vs this week", delta_color="inverse")
    c2.metric(f"This week ({weekly.index[-1]:%d %b})", f"Rs {base['price_now']:,.0f}/kg")
    c3.metric("ARIMA alone", f"Rs {base['arima_price']:,.0f}/kg")
    st.caption(f"Typical error on the 2025-26 test weeks: about Rs {rmse:.0f}/kg (RMSE), "
               f"Rs {scores['test_mae']:.0f}/kg on average (MAE). Treat the forecast as a direction and rough level, "
               "not an exact price.")

    recent = weekly["beans_pettah"].iloc[-26:].rename("Rs/kg").reset_index()
    point = pd.DataFrame({"date": [base["week"]], "Rs/kg": [base["price_next"]],
                          "low": [base["price_next"] - rmse], "high": [base["price_next"] + rmse]})
    joint = pd.DataFrame({"date": [weekly.index[-1], base["week"]],
                          "Rs/kg": [base["price_now"], base["price_next"]]})
    actual = alt.Chart(recent).mark_line(color=PETTAH, strokeWidth=2).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("Rs/kg:Q", title="Rs per kg",
                scale=alt.Scale(domain=[0, max(recent["Rs/kg"].max(), base["price_next"] + rmse) * 1.05])),
        tooltip=[alt.Tooltip("date:T", title="Week ending"), alt.Tooltip("Rs/kg:Q", format=",.0f")])
    link = alt.Chart(joint).mark_line(color=FORECAST, strokeDash=[4, 4]).encode(x="date:T", y="Rs/kg:Q")
    band = alt.Chart(point).mark_rule(color=FORECAST, strokeWidth=2).encode(x="date:T", y="low:Q", y2="high:Q")
    dot = alt.Chart(point).mark_circle(color=FORECAST, size=120, opacity=1).encode(
        x="date:T", y="Rs/kg:Q",
        tooltip=[alt.Tooltip("date:T", title="Forecast week"), alt.Tooltip("Rs/kg:Q", format=",.0f", title="Forecast")])
    st.altair_chart((actual + link + band + dot).properties(height=320), width="stretch")
    st.caption("Blue: actual weekly price (last 26 weeks). Green: next week's forecast, with a bar of ± one typical error.")

    st.subheader("What if this week had been different?")
    st.markdown("Change this week's numbers and see how the forecast responds. The earlier weeks stay as they are.")
    last = weekly.iloc[-1]
    with st.form("what_if"):
        c1, c2, c3 = st.columns(3)
        pettah = c1.number_input("Pettah beans (Rs/kg)", 50.0, 2000.0, float(last.beans_pettah), 10.0)
        dambulla = c1.number_input("Dambulla beans (Rs/kg)", 50.0, 2000.0, float(last.beans_dambulla), 10.0)
        carrot = c2.number_input("Pettah carrot (Rs/kg)", 20.0, 2000.0, float(last.carrot_pettah), 10.0)
        cabbage = c2.number_input("Pettah cabbage (Rs/kg)", 20.0, 2000.0, float(last.cabbage_pettah), 10.0)
        rain_ne = c3.number_input("Nuwara Eliya rain this week (mm)", 0.0, 1000.0,
                                  float(round(last.rain_nuwaraeliya, 1)), 5.0)
        rain_bd = c3.number_input("Badulla rain this week (mm)", 0.0, 1000.0, float(round(last.rain_badulla, 1)), 5.0)
        submitted = st.form_submit_button("Forecast")

    if submitted:
        changed = weekly.copy()
        changed.iloc[-1, changed.columns.get_indexer(
            ["beans_pettah", "beans_dambulla", "carrot_pettah", "cabbage_pettah", "rain_nuwaraeliya",
             "rain_badulla"])] = [pettah, dambulla, carrot, cabbage, rain_ne, rain_bd]
        fc = forecast_next_week(bundle, changed)
        c1, c2, c3 = st.columns(3)
        c1.metric("Your scenario: hybrid forecast", f"Rs {fc['price_next']:,.0f}/kg",
                  f"{fc['price_next'] - base['price_next']:+,.0f} vs the forecast above", delta_color="inverse")
        c2.metric("ARIMA part", f"Rs {fc['arima_price']:,.0f}/kg")
        c3.metric("Forest correction", f"{np.expm1(fc['rf_correction']):+.1%}",
                  help="How much the Random Forest moves ARIMA's forecast, using rain, other prices and the season.")

st.divider()
st.caption(f"Model trained on weeks up to {bundle['trained_until']}. Source code and notebook: "
           "github.com/renulucshmi/pettah-beans-price-forecast")
