# EX3_1_descriptives.py
# ============================================================
# Example 3.1
# Descriptive Statistics of Technology Indicators
#
# Dataset:
#   patent_trends.csv
#   columns: year, field, patents
#
# Outputs:
#   - summary_by_field.csv
#   - iqr_bounds.csv
#   - outliers.csv
#   - patent_indicators_processed.csv
#   - boxplot_patents_by_field.png
#   - line_patents_by_field.png
#   - line_zscore_by_field.png
#
# Note:
# This example uses synthetic patent-count data for
# educational and reproducibility purposes.
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import skew, kurtosis


# ============================================================
# 0) Setup
# ============================================================

DATA = "patent_trends.csv"

OUT_DIR = Path("outputs_ex3_1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) Load data and sanity checks
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA)

required_cols = {"year", "field", "patents"}

missing = required_cols - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# Ensure numeric data types
df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

df["patents"] = pd.to_numeric(
    df["patents"],
    errors="coerce"
)

# Check for invalid values
if df[["year", "patents"]].isna().any().any():
    raise ValueError(
        "Non-numeric values found in 'year' or 'patents'."
    )

# Sort and clean
df = (
    df.sort_values(["field", "year"])
      .drop_duplicates(subset=["year", "field"])
      .reset_index(drop=True)
)

# Ensure integer years
df["year"] = df["year"].astype(int)

print("\n=== First rows of dataset ===")
print(df.head())


# ============================================================
# 2) Summary statistics
# ============================================================

print("\nComputing summary statistics...")

summary = (
    df.groupby("field")["patents"]
      .agg(
          count="count",
          mean="mean",
          median="median",
          std="std",
          min="min",
          max="max",
          skewness=lambda x: skew(x, bias=False),
          kurtosis=lambda x: kurtosis(x, bias=False)
      )
      .assign(
          cv=lambda x: 100 * x["std"] / x["mean"]
      )
)

summary = summary.round(3)

summary.to_csv(
    OUT_DIR / "summary_by_field.csv",
    index=True
)

print("\n=== Summary Statistics by Field ===")
print(summary)


# ============================================================
# 3) IQR-based outlier detection
# ============================================================

print("\nDetecting outliers using IQR rule...")

q = (
    df.groupby("field")["patents"]
      .quantile([0.25, 0.75])
      .unstack()
)

q.columns = ["q1", "q3"]

iqr_df = q.assign(
    IQR=lambda x: x["q3"] - x["q1"]
)

iqr_df["lower"] = (
    iqr_df["q1"] - 1.5 * iqr_df["IQR"]
)

iqr_df["upper"] = (
    iqr_df["q3"] + 1.5 * iqr_df["IQR"]
)

iqr_df = iqr_df.round(3)

iqr_df.to_csv(
    OUT_DIR / "iqr_bounds.csv"
)

# Merge bounds into main dataframe
df = df.merge(
    iqr_df[["lower", "upper"]],
    left_on="field",
    right_index=True,
    how="left"
)

# Flag outliers
df["is_outlier"] = (
    (df["patents"] < df["lower"])
    |
    (df["patents"] > df["upper"])
)

# Save outliers separately
outliers = df[df["is_outlier"]].copy()

outliers.to_csv(
    OUT_DIR / "outliers.csv",
    index=False
)

print("\n=== Outliers ===")
print(outliers)


# ============================================================
# 4) Normalization
# ============================================================

print("\nApplying normalization methods...")

# -----------------------------
# Z-score normalization
# -----------------------------

def zscore(group):

    sd = group.std(ddof=1)

    if sd == 0:
        return np.zeros(len(group))

    return (group - group.mean()) / sd


# -----------------------------
# Min-max normalization
# -----------------------------

def minmax(group):

    rng = group.max() - group.min()

    if rng == 0:
        return np.zeros(len(group))

    return (group - group.min()) / rng


df["z_patents"] = (
    df.groupby("field")["patents"]
      .transform(zscore)
)

df["mm_patents"] = (
    df.groupby("field")["patents"]
      .transform(minmax)
)

# Save processed dataset
processed = df[
    [
        "year",
        "field",
        "patents",
        "z_patents",
        "mm_patents",
        "is_outlier"
    ]
]

processed = processed.round(3)

processed.to_csv(
    OUT_DIR / "patent_indicators_processed.csv",
    index=False
)

print("\n=== Processed Dataset ===")
print(processed.head())


# ============================================================
# 5) Boxplot visualization
# ============================================================

print("\nGenerating boxplot...")

fig, ax = plt.subplots(figsize=(8, 5))

fields = sorted(df["field"].unique())

data_to_plot = [
    df[df["field"] == f]["patents"]
    for f in fields
]

ax.boxplot(
    data_to_plot,
    tick_labels=fields
)

ax.set_title(
    "Annual Patent Counts by Technology Field (2010–2024)"
)

ax.set_xlabel("Technology Field")
ax.set_ylabel("Patent Count")

fig.tight_layout()

fig.savefig(
    OUT_DIR / "boxplot_patents_by_field.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 6) Line plot of patent trends
# ============================================================

print("\nGenerating patent trend plot...")

fig, ax = plt.subplots(figsize=(9, 5))

for field, group in df.groupby("field", sort=True):

    ax.plot(
        group["year"],
        group["patents"],
        marker="o",
        label=field
    )

ax.set_title(
    "Patent Trends by Technology Field (2010–2024)"
)

ax.set_xlabel("Year")
ax.set_ylabel("Patent Count")

ax.legend()
ax.grid(True)

fig.tight_layout()

fig.savefig(
    OUT_DIR / "line_patents_by_field.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 7) Z-score comparison plot
# ============================================================

print("\nGenerating z-score normalized comparison plot...")

fig, ax = plt.subplots(figsize=(9, 5))

for field, group in df.groupby("field", sort=True):

    ax.plot(
        group["year"],
        group["z_patents"],
        marker="o",
        label=field
    )

ax.set_title(
    "z-Score Normalized Patent Trends (2010–2024)"
)

ax.set_xlabel("Year")
ax.set_ylabel("Z-Score of Patents")

ax.legend()
ax.grid(True)

fig.tight_layout()

fig.savefig(
    OUT_DIR / "line_zscore_by_field.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 8) Final message
# ============================================================

print("\nDone.")
print("Outputs saved to:")
print(OUT_DIR.resolve())