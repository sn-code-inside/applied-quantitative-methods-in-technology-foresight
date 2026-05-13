# EX3_3_hypothesis.py
# ============================================================
# Example 3.3
# Hypothesis Testing Across Domains
# AI vs Robotics Annual Patent Growth Rates (2011–2024)
#
# Dataset:
#   patent_trends.csv
#   columns: year, field, patents
#
# Outputs:
#   - ex3_3_growth_rates.csv
#   - ex3_3_test_summary.json
#   - plot_growth_boxplot.png
#   - plot_growth_violin.png
#   - plot_growth_timeseries.png
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# 0) Setup
# ============================================================

DATA = "patent_trends.csv"

OUT = Path("outputs_ex3_3")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) Load and prepare data
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA)

required_cols = {"year", "field", "patents"}
missing = required_cols - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["patents"] = pd.to_numeric(df["patents"], errors="coerce")

if df[["year", "patents"]].isna().any().any():
    raise ValueError("Non-numeric values found in 'year' or 'patents'.")

df = (
    df[df["field"].isin(["AI", "Robotics"])]
    .sort_values(["field", "year"])
    .drop_duplicates(subset=["year", "field"])
    .reset_index(drop=True)
)

df["year"] = df["year"].astype(int)


# ============================================================
# 2) Compute annual growth rates
# ============================================================

print("\nComputing annual growth rates...")

df["growth"] = (
    df.groupby("field")["patents"]
    .pct_change()
)

growth_df = df.dropna(subset=["growth"]).copy()

growth_wide = growth_df.pivot(
    index="year",
    columns="field",
    values="growth"
).reset_index()

growth_wide.columns.name = None

growth_wide.to_csv(
    OUT / "ex3_3_growth_rates.csv",
    index=False
)

print("\n=== Growth Rates ===")
print(growth_wide.round(4))


# ============================================================
# 3) Extract groups
# ============================================================

g_ai = growth_df[growth_df["field"] == "AI"]["growth"]
g_rb = growth_df[growth_df["field"] == "Robotics"]["growth"]

n_ai = len(g_ai)
n_rb = len(g_rb)

mean_ai = g_ai.mean()
mean_rb = g_rb.mean()

sd_ai = g_ai.std(ddof=1)
sd_rb = g_rb.std(ddof=1)


# ============================================================
# 4) Assumption checks
# ============================================================

print("\nChecking test assumptions...")

# Shapiro-Wilk normality test
sh_ai = stats.shapiro(g_ai)
sh_rb = stats.shapiro(g_rb)

# Levene's test for equality of variances
lev = stats.levene(
    g_ai,
    g_rb,
    center="median"
)

equal_var = bool(lev.pvalue >= 0.05)


# ============================================================
# 5) Independent-samples t-test
# ============================================================

print("\nConducting independent-samples t-test...")

t_stat, p_val = stats.ttest_ind(
    g_ai,
    g_rb,
    equal_var=equal_var
)


# ============================================================
# 6) Effect size: Cohen's d
# ============================================================

s_pooled = np.sqrt(
    ((n_ai - 1) * sd_ai**2 + (n_rb - 1) * sd_rb**2)
    / (n_ai + n_rb - 2)
)

cohen_d = (mean_ai - mean_rb) / s_pooled


# ============================================================
# 7) Confidence interval for mean difference
# ============================================================

mean_diff = mean_ai - mean_rb

if equal_var:
    se_diff = s_pooled * np.sqrt(1 / n_ai + 1 / n_rb)
    df_t = n_ai + n_rb - 2
    test_type = "Pooled-variance t-test"
else:
    se_diff = np.sqrt(sd_ai**2 / n_ai + sd_rb**2 / n_rb)
    df_t = (
        (sd_ai**2 / n_ai + sd_rb**2 / n_rb) ** 2
        /
        (
            (sd_ai**2 / n_ai) ** 2 / (n_ai - 1)
            + (sd_rb**2 / n_rb) ** 2 / (n_rb - 1)
        )
    )
    test_type = "Welch's t-test"

crit = stats.t.ppf(0.975, df_t)

ci_low = mean_diff - crit * se_diff
ci_high = mean_diff + crit * se_diff


# ============================================================
# 8) Save test summary
# ============================================================

summary = {
    "sample_sizes": {
        "AI": int(n_ai),
        "Robotics": int(n_rb)
    },
    "descriptive_statistics": {
        "AI": {
            "mean_growth": float(mean_ai),
            "sd_growth": float(sd_ai)
        },
        "Robotics": {
            "mean_growth": float(mean_rb),
            "sd_growth": float(sd_rb)
        }
    },
    "assumption_checks": {
        "shapiro_AI": {
            "W": float(sh_ai.statistic),
            "p": float(sh_ai.pvalue)
        },
        "shapiro_Robotics": {
            "W": float(sh_rb.statistic),
            "p": float(sh_rb.pvalue)
        },
        "levene_test": {
            "statistic": float(lev.statistic),
            "p": float(lev.pvalue)
        }
    },
    "t_test": {
        "test_type": test_type,
        "equal_var_assumed": equal_var,
        "t_statistic": float(t_stat),
        "df": float(df_t),
        "p_value": float(p_val),
        "mean_difference_AI_minus_Robotics": float(mean_diff),
        "ci95_mean_difference": [
            float(ci_low),
            float(ci_high)
        ]
    },
    "effect_size": {
        "cohens_d": float(cohen_d)
    }
}

with open(
    OUT / "ex3_3_test_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n=== Test Summary ===")
print(json.dumps(summary, indent=2))


# ============================================================
# 9) Prepare long-format dataframe for visualization
# ============================================================

long_df = growth_df[["year", "field", "growth"]].copy()


# ============================================================
# 10) Boxplot
# ============================================================

print("\nGenerating boxplot...")

fig, ax = plt.subplots(figsize=(7, 5))

fields = ["AI", "Robotics"]

data_to_plot = [
    long_df[long_df["field"] == field]["growth"]
    for field in fields
]

ax.boxplot(
    data_to_plot,
    tick_labels=fields
)

ax.set_title("Annual Patent Growth Rates: AI vs Robotics")
ax.set_xlabel("Technology Field")
ax.set_ylabel("Growth Rate")

ax.axhline(0, linewidth=0.8)

fig.tight_layout()

fig.savefig(
    OUT / "plot_growth_boxplot.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 11) Violin plot
# ============================================================

print("\nGenerating violin plot...")

fig, ax = plt.subplots(figsize=(7, 5))

parts = ax.violinplot(
    data_to_plot,
    showmeans=True,
    showmedians=True
)

ax.set_xticks([1, 2])
ax.set_xticklabels(fields)

ax.set_title("Distribution of Annual Patent Growth Rates")
ax.set_xlabel("Technology Field")
ax.set_ylabel("Growth Rate")

ax.axhline(0, linewidth=0.8)

fig.tight_layout()

fig.savefig(
    OUT / "plot_growth_violin.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 12) Growth-rate time series plot
# ============================================================

print("\nGenerating growth-rate time series plot...")

fig, ax = plt.subplots(figsize=(9, 5))

for field, group in long_df.groupby("field", sort=True):
    ax.plot(
        group["year"],
        group["growth"],
        marker="o",
        label=field
    )

ax.set_title("Annual Patent Growth Rates Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("Growth Rate")

ax.axhline(0, linewidth=0.8)

ax.legend()
ax.grid(True)

fig.tight_layout()

fig.savefig(
    OUT / "plot_growth_timeseries.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 13) Final message
# ============================================================

print("\nDone.")
print("Outputs saved to:")
print(OUT.resolve())