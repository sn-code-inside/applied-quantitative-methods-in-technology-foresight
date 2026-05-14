"""
Panel GRU (multi-country) + Rolling-origin (expanding window) backtesting
Dataset: Our World in Data (CC-BY-4.0) — per-capita-energy-use.csv

Goal:
- Avoid the "single-country data scarcity" issue by training on ALL countries.
- Improve fairness vs a strong persistence baseline by:
  (1) Country-wise normalization (z-score per country)
  (2) Delta forecasting: predict change (delta) instead of absolute level
  (3) Temporal (time-based) validation inside each fold (no random validation_split)

Method:
- For each rolling-origin fold (cutoff_year):
  Train on all country data up to cutoff_year (expanding window).
  Validate on the most recent portion of training samples (temporal split).
  Test on cutoff_year+1 for countries that have observations.
- Baseline: delta = 0  (equivalent to persistence in normalized space)
- Model: GRU predicts delta using past LOOKBACK deltas + country embedding

Outputs:
- Average MAE/RMSE over folds for baseline vs GRU
- MAE/RMSE over time plots

------------------------------------------------------------
Reproducibility (for book chapter "Code Implementations"):
1) Fixed seeds (Python/NumPy/TF/Keras) + deterministic ops (best effort).
2) Stable ordering before sequence construction (sort by country_id, Year).
3) Time-based validation (no random split) to avoid leakage and randomness.
4) IMPORTANT: Exact environment versions used for reported results:

   - Python: 3.10.16 (Anaconda; MSC v.1929 64-bit, Windows)
   - NumPy: 2.0.1
   - Pandas: 2.2.3
   - TensorFlow: 2.18.1
   - Keras: 3.11.2
   - scikit-learn: 1.6.1

Notes on determinism:
- On GPU, some ops (including RNN kernels) may still be nondeterministic depending on
  CUDA/cuDNN and hardware. For strict reproducibility, prefer CPU runs and/or pin
  the exact CUDA/cuDNN stack (and ideally run in a container).
- The dataset is fetched from a URL; if upstream data changes, results can change.
  For archival reproducibility, download a local snapshot and load from disk.
------------------------------------------------------------
"""

# ============================
# 0) Reproducibility: set env vars BEFORE importing TF runtime
# ============================
import os
os.environ["PYTHONHASHSEED"] = "42"          # Stabilize hash-based operations
os.environ["TF_DETERMINISTIC_OPS"] = "1"     # Ask TF to prefer deterministic kernels where available
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"   # Best-effort deterministic behavior for cuDNN (GPU)

# Optional: reduce nondeterminism from parallelism (can slow down training).
# Uncomment if you still see variability across runs on the same machine.
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
# os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
import keras
from keras import layers

from pathlib import Path

# ============================
# 1) Configuration (single source of truth)
# ============================
URL = "https://ourworldindata.org/grapher/per-capita-energy-use.csv?v=1&csvType=full&useColumnShortNames=false"


PROJECT_NAME = "ch8_project_1"
OUTPUT_DIR = f"outputs_{PROJECT_NAME}"
# Reproducibility tip:
# - For a stable, archival run: download this CSV once and use a local file path.
#   Example: URL = "data/energy-use-per-capita_snapshot_2025-12-31.csv"

LOOKBACK = 3
MIN_HISTORY = 15
MIN_TEST_COUNTRIES = 50
MIN_TRAIN_SAMPLES = 2000

# Rolling-origin test range:
START_TEST_YEAR = None
END_TEST_YEAR = None

# Training settings (retrained per fold)
SEED = 42
EPOCHS = 60
BATCH_SIZE = 128
PATIENCE = 6
VAL_FRAC = 0.1

# Model hyperparameters
GRU_UNITS = 8
EMB_DIM = 2
DROPOUT = 0.0
REC_DROPOUT = 0.0

LOSS_NAME = "mae"            # or keras.losses.Huber(delta=1.0)
LR = 1e-3
CLIPNORM = 1.0

# ============================
# 2) Fix random seeds (NumPy + TF + Keras)
# ============================
# Reproducibility tip:
# - Set seeds ONCE at the top-level before any model building/training.
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Keras helper (TF/Keras supports this in newer versions; safe fallback otherwise)
try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass

# Best-effort deterministic ops at runtime (TF 2.18.1 supports this in most setups)
# NOTE: if an op has no deterministic implementation, TF may raise or silently fall back.
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

# ============================
# 3) Load and clean data
# ============================
# Reproducibility tip:
# - URL-based data can change over time. Pin a snapshot for published results.
df = pd.read_csv(
    URL,
    storage_options={"User-Agent": "Our World In Data data fetch/1.0"}
)

# Handle both possible formats:
# (1) Long format: Entity, Code, Year, value
# (2) Wide format: Entity, Code, years as columns
if "Year" not in df.columns:
    id_vars = [c for c in df.columns if c in ["Entity", "Code"]]
    value_vars = [c for c in df.columns if c not in id_vars]
    df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="Year", value_name="value")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Infer the value column name and standardize it
value_cols = [c for c in df.columns if c not in ["Entity", "Code", "Year"]]
if "energy_use_per_capita" not in df.columns:
    if len(value_cols) != 1:
        raise ValueError(f"Could not infer value column. Columns: {df.columns}")
    df = df.rename(columns={value_cols[0]: "energy_use_per_capita"})

# Basic cleaning (deterministic operations)
df = df.dropna(subset=["Entity", "Year", "energy_use_per_capita"]).copy()
df["Year"] = df["Year"].astype(int)
df["energy_use_per_capita"] = pd.to_numeric(df["energy_use_per_capita"], errors="coerce")
df = df.dropna(subset=["energy_use_per_capita"]).copy()

# Keep only countries with enough history
country_counts = df.groupby("Entity")["Year"].count()
eligible_countries = country_counts[country_counts >= MIN_HISTORY].index
df = df[df["Entity"].isin(eligible_countries)].copy()

print("Eligible countries:", df["Entity"].nunique())
print("Year range:", int(df["Year"].min()), "-", int(df["Year"].max()))
print(df.head())

# ============================
# 4) Encode countries (stable mapping)
# ============================
# Reproducibility tip:
# - Sorting ensures stable country_id assignment across runs.
countries = sorted(df["Entity"].unique().tolist())
country_to_id = {c: i for i, c in enumerate(countries)}
df["country_id"] = df["Entity"].map(country_to_id).astype(int)

num_countries = int(df["country_id"].nunique())
min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

# ============================
# 5) Country-wise normalization (z-score)
# ============================
stats = (
    df.groupby("country_id")["energy_use_per_capita"]
      .agg(["mean", "std"])
      .rename(columns={"mean": "mu", "std": "sigma"})
)

# Drop countries with zero std (flat series)
stats["sigma"] = stats["sigma"].replace(0, np.nan)
df = df.join(stats, on="country_id")
df = df.dropna(subset=["sigma"]).copy()

df["y_norm"] = ((df["energy_use_per_capita"] - df["mu"]) / df["sigma"]).astype(np.float32)

# Sort for reproducible sequence building
# Reproducibility tip:
# - Always sort before creating sequences; otherwise sample order may vary.
df = df.sort_values(["country_id", "Year"]).reset_index(drop=True)

print("Countries after removing zero-variance series:", int(df["country_id"].nunique()))

# ============================
# 6) Temporal validation split helper (time-ordered)
# ============================
def temporal_train_val_split(X, C, y, sample_year, val_frac=0.1):
    """
    Split training samples into train/validation using time order.
    Validation contains the most recent samples (largest sample_year).

    Reproducibility tip:
    - Use stable sorting (mergesort) to guarantee consistent ordering when ties exist.
    """
    sample_year = np.asarray(sample_year)
    order = np.argsort(sample_year, kind="mergesort")  # stable sort

    X = X[order]
    C = C[order]
    y = y[order]
    sample_year = sample_year[order]

    n = len(y)
    n_val = max(1, int(np.floor(n * val_frac)))
    n_train = n - n_val

    X_tr, C_tr, y_tr = X[:n_train], C[:n_train], y[:n_train]
    X_va, C_va, y_va = X[n_train:], C[n_train:], y[n_train:]

    return X_tr, C_tr, y_tr, X_va, C_va, y_va

# ============================
# 7) Fold builder for delta forecasting + sample_year for temporal validation
# ============================
def build_fold_delta(df_sorted: pd.DataFrame, lookback: int, cutoff_year: int):
    """
    Build train/test samples for a rolling-origin fold using DELTA forecasting in normalized space.

    Reproducibility tip:
    - Input df_sorted must be sorted by (country_id, Year).
    - Use sort=True in groupby for deterministic group ordering.
    """
    train_rows = df_sorted[df_sorted["Year"] <= cutoff_year]
    test_rows = df_sorted[df_sorted["Year"] == cutoff_year + 1]

    grouped_full = {cid: g.sort_values("Year") for cid, g in df_sorted.groupby("country_id", sort=True)}

    X_train, C_train, y_train, train_sample_year = [], [], [], []
    X_test, C_test, y_test, y_prev = [], [], [], []

    # Training samples
    for cid, g in train_rows.groupby("country_id", sort=True):
        g = g.sort_values("Year")
        years = g["Year"].to_numpy()
        y = g["y_norm"].to_numpy()

        if len(y) <= (lookback + 1):
            continue

        deltas = np.diff(y)  # delta index j corresponds to year years[j+1]

        for i in range(lookback, len(deltas)):
            X_train.append(deltas[i - lookback:i])
            y_train.append(deltas[i])
            C_train.append(cid)
            train_sample_year.append(int(years[i + 1]))

    # Test samples (countries with cutoff_year+1 observation)
    for cid, g_test in test_rows.groupby("country_id", sort=True):
        g_full = grouped_full[cid]
        g_hist = g_full[g_full["Year"] <= cutoff_year].sort_values("Year")

        if len(g_hist) <= (lookback + 1):
            continue

        y_hist = g_hist["y_norm"].to_numpy()
        deltas_hist = np.diff(y_hist)

        x = deltas_hist[-lookback:]
        y_prev_t = y_hist[-1]
        y_true_t1 = float(g_test["y_norm"].iloc[0])

        X_test.append(x)
        C_test.append(cid)
        y_test.append(y_true_t1)
        y_prev.append(y_prev_t)

    X_train = np.array(X_train, dtype=np.float32).reshape(-1, lookback, 1)
    y_train = np.array(y_train, dtype=np.float32)
    C_train = np.array(C_train, dtype=np.int32)
    train_sample_year = np.array(train_sample_year, dtype=np.int32)

    X_test = np.array(X_test, dtype=np.float32).reshape(-1, lookback, 1)
    y_test = np.array(y_test, dtype=np.float32)
    C_test = np.array(C_test, dtype=np.int32)
    y_prev = np.array(y_prev, dtype=np.float32)

    return X_train, C_train, y_train, train_sample_year, X_test, C_test, y_test, y_prev

# ============================
# 8) Panel GRU model for delta prediction
# ============================
def build_panel_gru_delta(lookback: int, n_countries: int, seed: int = 42) -> keras.Model:
    """
    Inputs:
      - seq: past LOOKBACK deltas (lookback x 1)
      - cid: country_id (embedded)
    Output:
      - predicted delta (normalized)

    Reproducibility tip:
    - Weight initialization is stochastic; seeds control initial weights.
    - On GPU, RNN kernels can still show tiny nondeterminism across runs.
    """
    # Keeping your behavior: reseed before building each fold model,
    # so each fold starts from the same initialization.
    tf.random.set_seed(seed)
    np.random.seed(seed)

    seq_in = layers.Input(shape=(lookback, 1), name="seq")
    cid_in = layers.Input(shape=(), dtype="int32", name="cid")

    x = layers.GRU(GRU_UNITS, dropout=DROPOUT, recurrent_dropout=REC_DROPOUT)(seq_in)

    emb = layers.Embedding(input_dim=n_countries, output_dim=EMB_DIM)(cid_in)
    emb = layers.Flatten()(emb)

    h = layers.Concatenate()([x, emb])
    h = layers.Dense(16, activation="relu")(h)
    out = layers.Dense(1)(h)

    model = keras.Model(inputs={"seq": seq_in, "cid": cid_in}, outputs=out)

    opt = keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM)
    model.compile(optimizer=opt, loss=LOSS_NAME)
    return model

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=0),
]

# ============================
# 9) Rolling-origin by year (expanding window)
# ============================
if START_TEST_YEAR is None:
    START_TEST_YEAR = min_year + (LOOKBACK + 5)
if END_TEST_YEAR is None:
    END_TEST_YEAR = max_year - 1

START_TEST_YEAR = max(START_TEST_YEAR, min_year + LOOKBACK + 1)
END_TEST_YEAR = min(END_TEST_YEAR, max_year - 1)

print(f"\nRolling-origin cutoff years: {START_TEST_YEAR}..{END_TEST_YEAR} (test years {START_TEST_YEAR+1}..{END_TEST_YEAR+1})")

fold_rows = []

for cutoff_year in range(START_TEST_YEAR, END_TEST_YEAR + 1):
    X_train, C_train, y_train, train_year, X_test, C_test, y_test, y_prev = build_fold_delta(df, LOOKBACK, cutoff_year)

    # Skip weak folds (deterministic thresholds)
    if len(X_test) < MIN_TEST_COUNTRIES or len(X_train) < MIN_TRAIN_SAMPLES:
        continue

    # Baseline in normalized space: delta=0 => yhat_norm = y_prev
    yhat_base_norm = y_prev

    # Temporal train/val split (validation = most recent samples)
    X_tr, C_tr, y_tr, X_va, C_va, y_va = temporal_train_val_split(
        X_train, C_train, y_train, train_year, val_frac=VAL_FRAC
    )

    model = build_panel_gru_delta(LOOKBACK, num_countries, seed=SEED)

    # Reproducibility tip:
    # - shuffle=False keeps batch order fixed; avoids extra randomness.
    model.fit(
        {"seq": X_tr, "cid": C_tr},
        y_tr,
        validation_data=({"seq": X_va, "cid": C_va}, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
        shuffle=False
    )

    # Predict delta and reconstruct normalized level forecasts
    delta_hat = model.predict({"seq": X_test, "cid": C_test}, verbose=0).reshape(-1)
    yhat_gru_norm = y_prev + delta_hat

    # Convert normalized levels back to kWh using country-specific mu/sigma
    mu = stats.loc[C_test, "mu"].to_numpy()
    sigma = stats.loc[C_test, "sigma"].to_numpy()

    y_true_kwh = (y_test * sigma + mu)
    y_base_kwh = (yhat_base_norm * sigma + mu)
    y_gru_kwh  = (yhat_gru_norm * sigma + mu)

    mae_base = mean_absolute_error(y_true_kwh, y_base_kwh)
    rmse_base = np.sqrt(mean_squared_error(y_true_kwh, y_base_kwh))

    mae_gru = mean_absolute_error(y_true_kwh, y_gru_kwh)
    rmse_gru = np.sqrt(mean_squared_error(y_true_kwh, y_gru_kwh))

    fold_rows.append({
        "cutoff_year": cutoff_year,
        "test_year": cutoff_year + 1,
        "n_test_countries": len(X_test),
        "MAE_base_kWh": mae_base,
        "RMSE_base_kWh": rmse_base,
        "MAE_gru_kWh": mae_gru,
        "RMSE_gru_kWh": rmse_gru,
    })

metrics_df = pd.DataFrame(fold_rows)

# ============================
# 10) Results: average over folds + plots
# ============================
print("\nFold-level metrics (head):")
print(metrics_df.head())

print("\nAverage over folds:")
print(pd.DataFrame({
    "Model": ["Baseline (delta=0)", "Panel GRU (delta)"],
    "MAE (kWh)": [metrics_df["MAE_base_kWh"].mean(), metrics_df["MAE_gru_kWh"].mean()],
    "RMSE (kWh)": [metrics_df["RMSE_base_kWh"].mean(), metrics_df["RMSE_gru_kWh"].mean()],
    "Folds": [len(metrics_df), len(metrics_df)],
    "Avg test countries/fold": [metrics_df["n_test_countries"].mean(), metrics_df["n_test_countries"].mean()],
}))

# Plot RMSE over time
if len(metrics_df) > 0:
    plt.figure()
    plt.plot(metrics_df["test_year"], metrics_df["RMSE_base_kWh"], label="Baseline RMSE")
    plt.plot(metrics_df["test_year"], metrics_df["RMSE_gru_kWh"], label="Panel GRU RMSE")
    plt.title("Rolling-origin RMSE over time (panel, delta, country-normalized, temporal validation)")
    plt.xlabel("Test year")
    plt.ylabel("RMSE (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot MAE over time
    plt.figure()
    plt.plot(metrics_df["test_year"], metrics_df["MAE_base_kWh"], label="Baseline MAE")
    plt.plot(metrics_df["test_year"], metrics_df["MAE_gru_kWh"], label="Panel GRU MAE")
    plt.title("Rolling-origin MAE over time (panel, delta, country-normalized, temporal validation)")
    plt.xlabel("Test year")
    plt.ylabel("MAE (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.show()

rmse_improve_rate = (metrics_df["RMSE_gru_kWh"] < metrics_df["RMSE_base_kWh"]).mean()
print("Share of folds where GRU improves RMSE:", rmse_improve_rate)

mae_improve_rate = (metrics_df["MAE_gru_kWh"] < metrics_df["MAE_base_kWh"]).mean()
print("Share of folds where GRU improves MAE:", mae_improve_rate)

# ----------------------------------
# Temporal Performance Patterns
# Figure 1: RMSE over time
# ----------------------------------
plt.figure(figsize=(8, 4))
plt.plot(metrics_df["test_year"], metrics_df["RMSE_base_kWh"], label="Baseline (delta = 0)", linewidth=2)
plt.plot(metrics_df["test_year"], metrics_df["RMSE_gru_kWh"], label="Panel GRU (delta)", linewidth=2)
plt.xlabel("Test year")
plt.ylabel("RMSE (kWh)")
plt.title("Temporal performance patterns: RMSE over rolling-origin test years")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------
# Figure 2: MAE over time
# ----------------------------------
plt.figure(figsize=(8, 4))
plt.plot(metrics_df["test_year"], metrics_df["MAE_base_kWh"], label="Baseline (delta = 0)", linewidth=2)
plt.plot(metrics_df["test_year"], metrics_df["MAE_gru_kWh"], label="Panel GRU (delta)", linewidth=2)
plt.xlabel("Test year")
plt.ylabel("MAE (kWh)")
plt.title("Temporal performance patterns: MAE over rolling-origin test years")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================
# 11) Save outputs
# ============================

os.makedirs(OUTPUT_DIR, exist_ok=True)

if len(metrics_df) == 0:
    raise ValueError(
        "No valid rolling-origin folds were produced. "
        "No output files were saved. "
        "Consider lowering MIN_TEST_COUNTRIES or MIN_TRAIN_SAMPLES."
    )

# File paths
metrics_path = os.path.join(OUTPUT_DIR, "rolling_origin_metrics.csv")
average_metrics_path = os.path.join(OUTPUT_DIR, "average_metrics.csv")
improvement_rates_path = os.path.join(OUTPUT_DIR, "improvement_rates.csv")
rmse_plot_path = os.path.join(OUTPUT_DIR, "rmse_over_time.png")
mae_plot_path = os.path.join(OUTPUT_DIR, "mae_over_time.png")

# Save fold-level metrics
metrics_df.to_csv(metrics_path, index=False)

# Save average metrics
average_metrics_df = pd.DataFrame({
    "Model": ["Baseline (delta=0)", "Panel GRU (delta)"],
    "MAE (kWh)": [
        metrics_df["MAE_base_kWh"].mean(),
        metrics_df["MAE_gru_kWh"].mean()
    ],
    "RMSE (kWh)": [
        metrics_df["RMSE_base_kWh"].mean(),
        metrics_df["RMSE_gru_kWh"].mean()
    ],
    "Folds": [
        len(metrics_df),
        len(metrics_df)
    ],
    "Avg test countries/fold": [
        metrics_df["n_test_countries"].mean(),
        metrics_df["n_test_countries"].mean()
    ],
})

average_metrics_df.to_csv(average_metrics_path, index=False)

# Save improvement rates
improvement_rates_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE"],
    "Share of folds where GRU improves": [
        (metrics_df["RMSE_gru_kWh"] < metrics_df["RMSE_base_kWh"]).mean(),
        (metrics_df["MAE_gru_kWh"] < metrics_df["MAE_base_kWh"]).mean()
    ]
})

improvement_rates_df.to_csv(improvement_rates_path, index=False)

# Save RMSE figure
plt.figure(figsize=(8, 4))
plt.plot(
    metrics_df["test_year"],
    metrics_df["RMSE_base_kWh"],
    label="Baseline (delta = 0)",
    linewidth=2
)
plt.plot(
    metrics_df["test_year"],
    metrics_df["RMSE_gru_kWh"],
    label="Panel GRU (delta)",
    linewidth=2
)
plt.xlabel("Test year")
plt.ylabel("RMSE (kWh)")
plt.title("Temporal performance patterns: RMSE over rolling-origin test years")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(rmse_plot_path, dpi=300, bbox_inches="tight")
plt.close()

# Save MAE figure
plt.figure(figsize=(8, 4))
plt.plot(
    metrics_df["test_year"],
    metrics_df["MAE_base_kWh"],
    label="Baseline (delta = 0)",
    linewidth=2
)
plt.plot(
    metrics_df["test_year"],
    metrics_df["MAE_gru_kWh"],
    label="Panel GRU (delta)",
    linewidth=2
)
plt.xlabel("Test year")
plt.ylabel("MAE (kWh)")
plt.title("Temporal performance patterns: MAE over rolling-origin test years")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(mae_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print("\nDone.")
print("Outputs saved to:")
print(os.path.abspath(OUTPUT_DIR))


# ============================
# 12) Reproducibility: exact environment versions (as reported)
# ============================
# Environment used for the reported experiments:
# Python: 3.10.16 | packaged by Anaconda, Inc. | (main, Dec 11 2024, 16:19:12) [MSC v.1929 64 bit (AMD64)]
# NumPy: 2.0.1
# Pandas: 2.2.3
# TensorFlow: 2.18.1
# Keras: 3.11.2
# scikit-learn: 1.6.1
#
# Optional runtime verification (uncomment if you want the script to print them):
# import sys, sklearn
# print("Python:", sys.version)
# print("NumPy:", np.__version__)
# print("Pandas:", pd.__version__)
# print("TensorFlow:", tf.__version__)
# print("Keras:", keras.__version__)
# print("scikit-learn:", sklearn.__version__)
