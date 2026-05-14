a# ch3_example_4_time_series.py
# ============================================================
# Example 3.4
# Time Series Forecasting of Publication Counts
#
# Dataset:
#   pub_trends.csv
#   columns: year, publications
#
# Outputs:
#   - adf_test.json
#   - trend_decomposition.png
#   - arima_test_forecast.csv
#   - ets_test_forecast.csv
#   - arima_future_forecast.csv
#   - ets_future_forecast.csv
#   - metrics_test.csv
#   - plot_test_forecasts.png
#   - plot_future_forecasts.png
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ============================================================
# 0) Setup and helper functions
# ============================================================

DATA = "pub_trends.csv"

OUT = Path("outputs_ch3_example_4")
OUT.mkdir(parents=True, exist_ok=True)


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    eps = 1e-12
    return float(100 * np.mean(np.abs((y - yhat) / np.maximum(np.abs(y), eps))))


def adf_summary(series, name):
    result = adfuller(series.dropna().values, autolag="AIC")

    return {
        "series": name,
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "used_lags": int(result[2]),
        "nobs": int(result[3]),
        "critical_values": {
            k: float(v) for k, v in result[4].items()
        },
        "icbest": float(result[5])
    }


# ============================================================
# 1) Load data
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA)

required_cols = {"year", "publications"}

missing = required_cols - set(df.columns)

if missing:
    raise ValueError(
        f"pub_trends.csv must contain columns: {missing}"
    )

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

df["publications"] = pd.to_numeric(
    df["publications"],
    errors="coerce"
)

df = (
    df.dropna(subset=["year", "publications"])
      .sort_values("year")
      .drop_duplicates(subset=["year"], keep="last")
      .reset_index(drop=True)
)

df["year"] = df["year"].astype(int)

if len(df) < 6:
    raise ValueError(
        "At least 6 annual observations are recommended for this workflow."
    )

print("\n=== Dataset ===")
print(df)


# Use PeriodIndex to avoid statsmodels unsupported-index warnings
period_index = pd.PeriodIndex(df["year"], freq="Y")
y = pd.Series(
    df["publications"].astype(float).values,
    index=period_index,
    name="publications"
)


# ============================================================
# 2) Stationarity: ADF tests
# ============================================================

print("\nRunning ADF tests...")

adf_results = [
    adf_summary(y, "Publications_Level"),
    adf_summary(y.diff(), "Publications_First_Difference")
]

with open(
    OUT / "adf_test.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(adf_results, f, indent=2)

print("\n=== ADF Test Results ===")
print(json.dumps(adf_results, indent=2))


# ============================================================
# 3) Trend decomposition using moving average
# ============================================================

print("\nGenerating trend decomposition plot...")

# Annual data usually has no meaningful seasonality.
# Therefore, we use a 3-year centered moving average
# to extract the trend component.

trend = y.rolling(
    window=3,
    center=True,
    min_periods=1
).mean()

residual = y - trend

years = df["year"].values

fig, axes = plt.subplots(
    3,
    1,
    figsize=(8, 6),
    sharex=True
)

axes[0].plot(
    years,
    y.values,
    marker="o"
)

axes[0].set_title("Observed Series")
axes[0].set_ylabel("Publications")

axes[1].plot(
    years,
    trend.values,
    marker="o"
)

axes[1].set_title("Trend Component (3-Year Moving Average)")
axes[1].set_ylabel("Trend")

axes[2].plot(
    years,
    residual.values,
    marker="o"
)

axes[2].axhline(0, linewidth=0.8)
axes[2].set_title("Residual Component")
axes[2].set_xlabel("Year")
axes[2].set_ylabel("Residual")

fig.tight_layout()

fig.savefig(
    OUT / "trend_decomposition.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 4) Train / test split
# ============================================================

print("\nPreparing train/test split...")

# Train: 2010–2022
# Test: 2023–2024

train = y[y.index.year <= 2022]
test = y[y.index.year >= 2023]

if len(train) < 5:
    raise ValueError(
        "Training set is too short. Check that data include enough years before 2023."
    )

if len(test) == 0:
    raise ValueError(
        "Test set is empty. Check that data include 2023–2024 observations."
    )

h_test = len(test)
h_future = 5

test_years = test.index.year


# ============================================================
# 5) Fit models on TRAIN and evaluate on TEST
# ============================================================

print("\nFitting ARIMA and ETS models on training data...")

# -----------------------------
# ARIMA(1,1,1)
# -----------------------------

arima_train = ARIMA(
    train,
    order=(1, 1, 1)
).fit()

arima_test_forecast = arima_train.forecast(
    steps=h_test
)

arima_test_df = pd.DataFrame({
    "year": test_years,
    "actual": test.values,
    "forecast": arima_test_forecast.values
})

arima_test_df.to_csv(
    OUT / "arima_test_forecast.csv",
    index=False
)


# -----------------------------
# ETS-like model
# Additive trend, no seasonality
# -----------------------------

ets_train = ExponentialSmoothing(
    train,
    trend="add",
    seasonal=None,
    initialization_method="estimated"
).fit()

ets_test_forecast = ets_train.forecast(
    steps=h_test
)

ets_test_df = pd.DataFrame({
    "year": test_years,
    "actual": test.values,
    "forecast": ets_test_forecast.values
})

ets_test_df.to_csv(
    OUT / "ets_test_forecast.csv",
    index=False
)


# ============================================================
# 6) Forecast accuracy metrics
# ============================================================

print("\nComputing forecast accuracy metrics...")

metrics = pd.DataFrame([
    {
        "model": "ARIMA(1,1,1)",
        "MAE": mae(test.values, arima_test_forecast.values),
        "RMSE": rmse(test.values, arima_test_forecast.values),
        "MAPE": mape(test.values, arima_test_forecast.values)
    },
    {
        "model": "ETS(Additive Trend)",
        "MAE": mae(test.values, ets_test_forecast.values),
        "RMSE": rmse(test.values, ets_test_forecast.values),
        "MAPE": mape(test.values, ets_test_forecast.values)
    }
])

metrics = metrics.round(4)

metrics.to_csv(
    OUT / "metrics_test.csv",
    index=False
)

print("\n=== Forecast Accuracy Metrics ===")
print(metrics)


# ============================================================
# 7) Plot test forecasts vs actuals
# ============================================================

print("\nGenerating test forecast plot...")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    train.index.year,
    train.values,
    marker="o",
    label="Train"
)

ax.plot(
    test.index.year,
    test.values,
    marker="o",
    label="Actual (Test)"
)

ax.plot(
    test.index.year,
    arima_test_forecast.values,
    marker="o",
    label="ARIMA(1,1,1) Forecast"
)

ax.plot(
    test.index.year,
    ets_test_forecast.values,
    marker="o",
    label="ETS Forecast"
)

ax.set_title("Test Forecasts vs Actual (Energy Publications)")
ax.set_xlabel("Year")
ax.set_ylabel("Publications")

ax.legend()
ax.grid(True)

fig.tight_layout()

fig.savefig(
    OUT / "plot_test_forecasts.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 8) Refit on FULL data and produce 5-year-ahead forecasts
# ============================================================

print("\nRefitting models on full data and forecasting future values...")

arima_full = ARIMA(
    y,
    order=(1, 1, 1)
).fit()

ets_full = ExponentialSmoothing(
    y,
    trend="add",
    seasonal=None,
    initialization_method="estimated"
).fit()

arima_future = arima_full.forecast(
    steps=h_future
)

ets_future = ets_full.forecast(
    steps=h_future
)

future_years = np.arange(
    int(df["year"].max()) + 1,
    int(df["year"].max()) + 1 + h_future
)

arima_future_df = pd.DataFrame({
    "year": future_years,
    "forecast": arima_future.values
}).round(4)

ets_future_df = pd.DataFrame({
    "year": future_years,
    "forecast": ets_future.values
}).round(4)

arima_future_df.to_csv(
    OUT / "arima_future_forecast.csv",
    index=False
)

ets_future_df.to_csv(
    OUT / "ets_future_forecast.csv",
    index=False
)

future_forecast_df = pd.DataFrame({
    "year": future_years,
    "ARIMA_forecast": arima_future.values,
    "ETS_forecast": ets_future.values
}).round(4)

print("\n=== Future Forecasts ===")
print(future_forecast_df)


# ============================================================
# 9) Plot historical + future forecasts
# ============================================================

print("\nGenerating future forecast plot...")

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    y.index.year,
    y.values,
    marker="o",
    label="Historical"
)

ax.plot(
    future_years,
    arima_future.values,
    marker="o",
    label="ARIMA(1,1,1) Forecast (5y)"
)

ax.plot(
    future_years,
    ets_future.values,
    marker="o",
    label="ETS Forecast (5y)"
)

ax.set_title("5-Year Ahead Forecasts — Energy Publications")
ax.set_xlabel("Year")
ax.set_ylabel("Publications")

ax.legend()
ax.grid(True)

fig.tight_layout()

fig.savefig(
    OUT / "plot_future_forecasts.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 10) Final message
# ============================================================

print("\nDone.")
print("Outputs saved to:")
print(OUT.resolve())
