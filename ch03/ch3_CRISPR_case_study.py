a# ch3_CRISPR_case_study.py
# ============================================================
# Case Study
# Retrospective Statistical Analysis of CRISPR Gene Editing
# Technology Evolution (2012–2024)
#
# This script uses assumed/synthetic sample tables designed
# to reproduce the statistical workflow described in the
# case study. These are not authoritative real-world counts.
#
# Outputs:
#   - table_3_1_publications_patents.csv
#   - table_3_2_regional_summary.csv
#   - table_3_3_clinical_trials.csv
#   - table_3_3_clinical_trials_summary.csv
#   - table_3_4_arima_forecasts.csv
#   - s_curve_results.csv
#   - hypothesis_test_results.csv
#   - anova_clinical_trials.csv
#   - adf_results.json
#   - summary_foresight_findings.csv
#   - several PNG plots
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy import stats

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


# ============================================================
# 0) Setup and helper functions
# ============================================================

OUT = Path("outputs_ch3_CRISPR_case_study")
OUT.mkdir(parents=True, exist_ok=True)


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)

    sp = np.sqrt(
        ((nx - 1) * sx**2 + (ny - 1) * sy**2)
        / (nx + ny - 2)
    )

    return (np.mean(x) - np.mean(y)) / sp


def logistic(t, K, r, t0):
    return K / (1.0 + np.exp(-r * (t - t0)))


def r_squared(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot


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
# 1) Table 3.1: Publications and patents
# ============================================================

print("\nCreating Table 3.1...")

df = pd.DataFrame({
    "Year": list(range(2012, 2025)),
    "Publications": [
        234, 891, 2103, 4567, 7234, 9891, 11567,
        12345, 12891, 13234, 13456, 13567, 13623
    ],
    "Patents": [
        12, 45, 124, 287, 523, 876, 1234,
        1567, 1789, 1923, 1998, 2034, 2056
    ]
})

df["Pub_YoY_%"] = df["Publications"].pct_change() * 100
df["Pat_YoY_%"] = df["Patents"].pct_change() * 100

df["Pub_MA3"] = (
    df["Publications"]
    .rolling(window=3, min_periods=1)
    .mean()
)

df["Pat_MA3"] = (
    df["Patents"]
    .rolling(window=3, min_periods=1)
    .mean()
)

table_3_1 = df.round(3)

table_3_1.to_csv(
    OUT / "table_3_1_publications_patents.csv",
    index=False
)

print("\n=== Table 3.1 ===")
print(table_3_1.to_string(index=False))


# ============================================================
# 2) Descriptive plots
# ============================================================

print("\nGenerating descriptive plots...")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(df["Year"], df["Publications"], marker="o", label="Publications")
ax.plot(df["Year"], df["Patents"], marker="o", label="Patents")

ax.set_title("CRISPR Publications and Patents (2012–2024)")
ax.set_xlabel("Year")
ax.set_ylabel("Count")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_publications_patents.png", dpi=200)
plt.show()
plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(df["Year"], df["Pub_YoY_%"], marker="o", label="Publication YoY Growth")
ax.plot(df["Year"], df["Pat_YoY_%"], marker="o", label="Patent YoY Growth")

ax.axhline(0, linewidth=0.8)

ax.set_title("Year-over-Year Growth Rates")
ax.set_xlabel("Year")
ax.set_ylabel("Growth Rate (%)")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_yoy_growth.png", dpi=200)
plt.show()
plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(df["Year"], df["Publications"], marker="o", alpha=0.5, label="Publications")
ax.plot(df["Year"], df["Pub_MA3"], marker="o", label="Publications MA(3)")

ax.set_title("CRISPR Publications with 3-Year Moving Average")
ax.set_xlabel("Year")
ax.set_ylabel("Publications")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_publications_ma3.png", dpi=200)
plt.show()
plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(df["Year"], df["Patents"], marker="o", alpha=0.5, label="Patents")
ax.plot(df["Year"], df["Pat_MA3"], marker="o", label="Patents MA(3)")

ax.set_title("CRISPR Patents with 3-Year Moving Average")
ax.set_xlabel("Year")
ax.set_ylabel("Patents")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_patents_ma3.png", dpi=200)
plt.show()
plt.close(fig)


# ============================================================
# 3) Logistic S-curve fitting
# ============================================================

print("\nFitting logistic S-curves...")

t = np.arange(len(df))

pub_y = df["Publications"].values.astype(float)
pat_y = df["Patents"].values.astype(float)

params_pub, _ = curve_fit(
    logistic,
    t,
    pub_y,
    p0=[pub_y.max() * 1.05, 0.5, len(t) / 2],
    maxfev=10000
)

K_pub, r_pub, t0_pub = params_pub
pub_hat = logistic(t, *params_pub)

r2_pub = r_squared(pub_y, pub_hat)
inflection_pub = df.loc[0, "Year"] + t0_pub

params_pat, _ = curve_fit(
    logistic,
    t,
    pat_y,
    p0=[pat_y.max() * 1.05, 0.5, len(t) / 2],
    maxfev=10000
)

K_pat, r_pat, t0_pat = params_pat
pat_hat = logistic(t, *params_pat)

r2_pat = r_squared(pat_y, pat_hat)
inflection_pat = df.loc[0, "Year"] + t0_pat

s_curve_results = pd.DataFrame([
    {
        "Indicator": "Publications",
        "K": K_pub,
        "r": r_pub,
        "t0_index": t0_pub,
        "Inflection_Year": inflection_pub,
        "R_squared": r2_pub
    },
    {
        "Indicator": "Patents",
        "K": K_pat,
        "r": r_pat,
        "t0_index": t0_pat,
        "Inflection_Year": inflection_pat,
        "R_squared": r2_pat
    }
]).round(4)

s_curve_results.to_csv(
    OUT / "s_curve_results.csv",
    index=False
)

print("\n=== Logistic S-Curve Results ===")
print(s_curve_results.to_string(index=False))


t_dense = np.linspace(t.min(), t.max(), 300)
years_dense = df.loc[0, "Year"] + t_dense

fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(df["Year"], pub_y, label="Observed Publications")
ax.plot(years_dense, logistic(t_dense, *params_pub), label="Logistic Fit")
ax.axvline(
    inflection_pub,
    linestyle="--",
    label=f"Inflection ≈ {inflection_pub:.1f}"
)

ax.set_title("Logistic S-Curve Fit: Publications")
ax.set_xlabel("Year")
ax.set_ylabel("Publications")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_s_curve_publications.png", dpi=200)
plt.show()
plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(df["Year"], pat_y, label="Observed Patents")
ax.plot(years_dense, logistic(t_dense, *params_pat), label="Logistic Fit")
ax.axvline(
    inflection_pat,
    linestyle="--",
    label=f"Inflection ≈ {inflection_pat:.1f}"
)

ax.set_title("Logistic S-Curve Fit: Patents")
ax.set_xlabel("Year")
ax.set_ylabel("Patents")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_s_curve_patents.png", dpi=200)
plt.show()
plt.close(fig)


# ============================================================
# 4) Table 3.2: Regional summary statistics
# ============================================================

print("\nCreating Table 3.2...")

years_late = np.array([2020, 2021, 2022, 2023, 2024])

us_pubs = np.array([3110, 3340, 3255, 3475, 2990])
cn_pubs = np.array([2605, 2910, 3010, 3190, 2740])

cn_pat = np.array([520, 560, 540, 505, 545])
us_pat = np.array([470, 520, 455, 500, 490])

table_3_2 = pd.DataFrame([
    {
        "Indicator": "Publications",
        "Region": "US",
        "Mean": us_pubs.mean(),
        "SD": us_pubs.std(ddof=1),
        "n": len(us_pubs)
    },
    {
        "Indicator": "Publications",
        "Region": "China",
        "Mean": cn_pubs.mean(),
        "SD": cn_pubs.std(ddof=1),
        "n": len(cn_pubs)
    },
    {
        "Indicator": "Patents",
        "Region": "US",
        "Mean": us_pat.mean(),
        "SD": us_pat.std(ddof=1),
        "n": len(us_pat)
    },
    {
        "Indicator": "Patents",
        "Region": "China",
        "Mean": cn_pat.mean(),
        "SD": cn_pat.std(ddof=1),
        "n": len(cn_pat)
    }
]).round(3)

table_3_2.to_csv(
    OUT / "table_3_2_regional_summary.csv",
    index=False
)

print("\n=== Table 3.2 ===")
print(table_3_2.to_string(index=False))


# ============================================================
# 5) Hypothesis tests
# ============================================================

print("\nConducting hypothesis tests...")

t_pub, p_pub_two = stats.ttest_ind(
    us_pubs,
    cn_pubs,
    equal_var=False
)

p_pub_one = p_pub_two / 2 if t_pub > 0 else 1 - p_pub_two / 2
d_pub = cohens_d(us_pubs, cn_pubs)

t_pat, p_pat_two = stats.ttest_ind(
    cn_pat,
    us_pat,
    equal_var=False
)

p_pat_one = p_pat_two / 2 if t_pat > 0 else 1 - p_pat_two / 2
d_pat = cohens_d(cn_pat, us_pat)

hypothesis_results = pd.DataFrame([
    {
        "Test": "US publications > China publications",
        "t_statistic": t_pub,
        "p_value_one_tailed": p_pub_one,
        "Cohens_d": d_pub
    },
    {
        "Test": "China patents > US patents",
        "t_statistic": t_pat,
        "p_value_one_tailed": p_pat_one,
        "Cohens_d": d_pat
    }
]).round(4)

hypothesis_results.to_csv(
    OUT / "hypothesis_test_results.csv",
    index=False
)

print("\n=== Hypothesis Test Results ===")
print(hypothesis_results.to_string(index=False))


# ============================================================
# 6) Table 3.3: Clinical trials and ANOVA
# ============================================================

print("\nCreating Table 3.3...")

us_trials = np.array([12, 14, 13, 15, 13])
cn_trials = np.array([6, 7, 8, 7, 6])
eu_trials = np.array([5, 6, 6, 5, 6])

clinical_df = pd.DataFrame({
    "Year": np.tile(years_late, 3),
    "Region": ["US"] * 5 + ["China"] * 5 + ["EU"] * 5,
    "Clinical_Trials": np.concatenate([us_trials, cn_trials, eu_trials])
})

clinical_df.to_csv(
    OUT / "table_3_3_clinical_trials.csv",
    index=False
)

clinical_summary = (
    clinical_df.groupby("Region")["Clinical_Trials"]
    .agg(Total="sum", Mean="mean", SD="std", n="count")
    .reset_index()
    .round(3)
)

clinical_summary.to_csv(
    OUT / "table_3_3_clinical_trials_summary.csv",
    index=False
)

F_stat, p_anova = stats.f_oneway(
    us_trials,
    cn_trials,
    eu_trials
)

anova_results = pd.DataFrame([{
    "F_statistic": F_stat,
    "p_value": p_anova
}]).round(4)

anova_results.to_csv(
    OUT / "anova_clinical_trials.csv",
    index=False
)

print("\n=== Table 3.3 ===")
print(clinical_summary.to_string(index=False))

print("\n=== ANOVA Results ===")
print(anova_results.to_string(index=False))


fig, ax = plt.subplots(figsize=(8, 5))

for region, group in clinical_df.groupby("Region"):
    ax.plot(
        group["Year"],
        group["Clinical_Trials"],
        marker="o",
        label=region
    )

ax.set_title("Clinical Trial Counts by Region (2020–2024)")
ax.set_xlabel("Year")
ax.set_ylabel("Clinical Trials")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_clinical_trials_by_region.png", dpi=200)
plt.show()
plt.close(fig)


# ============================================================
# 7) Table 3.4: ADF tests and ARIMA forecasting
# ============================================================

print("\nCreating Table 3.4...")

period_index = pd.PeriodIndex(df["Year"], freq="Y")

pub_series = pd.Series(
    df["Publications"].values,
    index=period_index,
    name="Publications"
)

pat_series = pd.Series(
    df["Patents"].values,
    index=period_index,
    name="Patents"
)

adf_results = [
    adf_summary(pub_series, "Publications_Level"),
    adf_summary(pub_series.diff(), "Publications_First_Difference"),
    adf_summary(pat_series, "Patents_Level"),
    adf_summary(pat_series.diff(), "Patents_First_Difference")
]

with open(
    OUT / "adf_results.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(adf_results, f, indent=2)

print("\n=== ADF Results ===")
print(json.dumps(adf_results, indent=2))


pub_model = ARIMA(
    pub_series,
    order=(1, 1, 1)
).fit()

pat_model = ARIMA(
    pat_series,
    order=(2, 1, 1)
).fit()

forecast_years = np.array([2025, 2026, 2027])

pub_forecast = pub_model.get_forecast(steps=3)
pat_forecast = pat_model.get_forecast(steps=3)

pub_mean = pub_forecast.predicted_mean
pub_ci = pub_forecast.conf_int(alpha=0.05)

pat_mean = pat_forecast.predicted_mean
pat_ci = pat_forecast.conf_int(alpha=0.05)

table_3_4 = pd.DataFrame({
    "Year": forecast_years,
    "Publication_Forecast": pub_mean.values,
    "Publication_Lower95": pub_ci.iloc[:, 0].values,
    "Publication_Upper95": pub_ci.iloc[:, 1].values,
    "Patent_Forecast": pat_mean.values,
    "Patent_Lower95": pat_ci.iloc[:, 0].values,
    "Patent_Upper95": pat_ci.iloc[:, 1].values
}).round(3)

table_3_4.to_csv(
    OUT / "table_3_4_arima_forecasts.csv",
    index=False
)

print("\n=== Table 3.4 ===")
print(table_3_4.to_string(index=False))


fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    pub_series.index.year,
    pub_series.values,
    marker="o",
    label="Observed Publications"
)

ax.plot(
    forecast_years,
    pub_mean.values,
    marker="o",
    label="Forecast"
)

ax.fill_between(
    forecast_years,
    pub_ci.iloc[:, 0].values,
    pub_ci.iloc[:, 1].values,
    alpha=0.2,
    label="95% Prediction Interval"
)

ax.set_title("Publication Forecasts (2025–2027)")
ax.set_xlabel("Year")
ax.set_ylabel("Publications")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_publication_forecast.png", dpi=200)
plt.show()
plt.close(fig)


fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    pat_series.index.year,
    pat_series.values,
    marker="o",
    label="Observed Patents"
)

ax.plot(
    forecast_years,
    pat_mean.values,
    marker="o",
    label="Forecast"
)

ax.fill_between(
    forecast_years,
    pat_ci.iloc[:, 0].values,
    pat_ci.iloc[:, 1].values,
    alpha=0.2,
    label="95% Prediction Interval"
)

ax.set_title("Patent Forecasts (2025–2027)")
ax.set_xlabel("Year")
ax.set_ylabel("Patents")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUT / "plot_patent_forecast.png", dpi=200)
plt.show()
plt.close(fig)


# ============================================================
# 8) Integrated foresight summary
# ============================================================

summary = pd.DataFrame([
    {
        "Analysis": "Maturation trajectory",
        "Finding": (
            "Publications and patents show rapid early expansion, "
            "decelerating growth, and later stabilization."
        )
    },
    {
        "Analysis": "S-curve modeling",
        "Finding": (
            "Logistic growth fits indicate saturation behavior and "
            "inflection points associated with lifecycle transition."
        )
    },
    {
        "Analysis": "Regional leadership",
        "Finding": (
            "The US remains stronger in publications, while China shows "
            "stronger patent activity in the late-period synthetic sample."
        )
    },
    {
        "Analysis": "Clinical translation",
        "Finding": (
            "Clinical trial counts remain highest in the US, indicating "
            "concentrated translational capacity."
        )
    },
    {
        "Analysis": "Forecasting",
        "Finding": (
            "Near-term ARIMA forecasts suggest continued stabilization "
            "and only marginal growth through 2027."
        )
    }
])

summary.to_csv(
    OUT / "summary_foresight_findings.csv",
    index=False
)

print("\nDone.")
print("Outputs saved to:")
print(OUT.resolve())
