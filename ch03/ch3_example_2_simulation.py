# ch3_example_2_simulation.py
# ============================================================
# Example 3.2
# Monte Carlo Simulation of Technology Adoption
# (Bass-type diffusion model)
#
# Dataset:
#   innovation_adoption_sim.csv
#
# Outputs:
#   - adoption_percentiles.csv
#   - adoption_summary.csv
#   - simulated_paths_sample.csv
#   - adoption_fanchart.png
#   - adoption_sample_paths.png
#   - params_used.json
#
# Note:
# This example uses a Monte Carlo simulation framework
# based on a Bass-type diffusion process to illustrate
# uncertainty in technology adoption trajectories.
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from pathlib import Path


# ============================================================
# 0) Reproducibility & output folder
# ============================================================

np.random.seed(42)

OUT_DIR = Path("outputs_ch3_example_2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) Bass diffusion simulation setup
# ============================================================

print("\nInitializing simulation settings...")

T = 15          # Number of periods
N = 10000       # Number of Monte Carlo paths
M = 1.0         # Market potential (normalized adoption share)

# Beta priors for innovation and imitation parameters
a_p, b_p = 2, 20
a_q, b_q = 4, 10

# Draw innovation (p) and imitation (q) parameters
p = np.random.beta(a_p, b_p, N)
q = np.random.beta(a_q, b_q, N)

print(f"\nSimulation horizon: {T} periods")
print(f"Monte Carlo paths: {N}")


# ============================================================
# 2) Simulate adoption paths
# ============================================================

print("\nGenerating Monte Carlo adoption trajectories...")

# adopt[n, t] = cumulative adoption share at period t
adopt = np.zeros((N, T))

for t in range(T):

    prev = adopt[:, t - 1] if t > 0 else np.zeros(N)

    # Bass-type increment
    dA = (
        (p + q * (prev / M))
        * (M - prev)
    )

    curr = prev + dA

    # Numerical safety
    curr = np.clip(curr, 0.0, M)

    adopt[:, t] = curr


# ============================================================
# 3) Percentile summary
# ============================================================

print("\nComputing percentile summaries...")

# Percentiles:
# 2.5–97.5 => 95% uncertainty band
# 10–90    => 80% uncertainty band
# 50       => median trajectory

percentiles = np.percentile(
    adopt,
    q=[2.5, 10, 50, 90, 97.5],
    axis=0
)

percentiles_df = pd.DataFrame({
    "year": np.arange(1, T + 1),
    "p2_5": percentiles[0],
    "p10": percentiles[1],
    "p50": percentiles[2],
    "p90": percentiles[3],
    "p97_5": percentiles[4],
})

percentiles_df = percentiles_df.round(4)

percentiles_df.to_csv(
    OUT_DIR / "adoption_percentiles.csv",
    index=False
)

print("\n=== Percentile Summary ===")
print(percentiles_df)


# ============================================================
# 4) Summary statistics by year
# ============================================================

print("\nComputing yearly summary statistics...")

summary_df = pd.DataFrame({
    "year": np.arange(1, T + 1),
    "mean": adopt.mean(axis=0),
    "std": adopt.std(axis=0),
    "min": adopt.min(axis=0),
    "max": adopt.max(axis=0),
})

summary_df = summary_df.round(4)

summary_df.to_csv(
    OUT_DIR / "adoption_summary.csv",
    index=False
)

print("\n=== Adoption Summary Statistics ===")
print(summary_df)


# ============================================================
# 5) Save sample trajectories
# ============================================================

print("\nSaving sample adoption trajectories...")

sample_paths = pd.DataFrame(
    adopt[:50].T
)

sample_paths.index = np.arange(1, T + 1)
sample_paths.index.name = "year"

sample_paths = sample_paths.round(4)

sample_paths.to_csv(
    OUT_DIR / "simulated_paths_sample.csv"
)


# ============================================================
# 6) Save simulation parameters
# ============================================================

print("\nSaving simulation parameters...")

params = {
    "seed": 42,
    "T": T,
    "N": N,
    "M": M,
    "a_p": a_p,
    "b_p": b_p,
    "a_q": a_q,
    "b_q": b_q,
}

with open(
    OUT_DIR / "params_used.json",
    "w"
) as f:
    json.dump(params, f, indent=2)


# ============================================================
# 7) Fan chart visualization
# ============================================================

print("\nGenerating fan chart...")

years = percentiles_df["year"].values

fig, ax = plt.subplots(figsize=(9, 5))

# Median trajectory
ax.plot(
    years,
    percentiles_df["p50"],
    linewidth=2,
    label="Median (50th percentile)"
)

# 80% uncertainty band
ax.fill_between(
    years,
    percentiles_df["p10"],
    percentiles_df["p90"],
    alpha=0.35,
    label="80% uncertainty band"
)

# 95% uncertainty band
ax.fill_between(
    years,
    percentiles_df["p2_5"],
    percentiles_df["p97_5"],
    alpha=0.20,
    label="95% uncertainty band"
)

ax.set_title(
    "Monte Carlo Simulation of Technology Adoption"
)

ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Adoption Share")

ax.set_ylim(0, 1.0)

ax.legend()
ax.grid(True)

fig.tight_layout()

fig.savefig(
    OUT_DIR / "adoption_fanchart.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 8) Plot sample adoption trajectories
# ============================================================

print("\nGenerating sample trajectory plot...")

fig, ax = plt.subplots(figsize=(9, 5))

for i in range(50):

    ax.plot(
        np.arange(1, T + 1),
        adopt[i],
        alpha=0.15
    )

ax.set_title(
    "Sample Monte Carlo Adoption Trajectories"
)

ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Adoption Share")

ax.set_ylim(0, 1.0)

ax.grid(True)

fig.tight_layout()

fig.savefig(
    OUT_DIR / "adoption_sample_paths.png",
    dpi=200
)

plt.show()
plt.close(fig)


# ============================================================
# 9) Final message
# ============================================================

print("\nDone.")
print("Outputs saved to:")
print(OUT_DIR.resolve())
