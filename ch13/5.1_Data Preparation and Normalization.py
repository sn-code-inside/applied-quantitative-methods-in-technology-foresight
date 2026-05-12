# ==============================================================================
# Exercise 5.1: Data Preparation and Normalization for Temporal Analysis
# ==============================================================================
#
# Before applying any statistical model, data must be normalized to ensure
# that the detected growth reflects the term's relative importance and not
# just the overall growth of the scientific domain.
#
# Objective: Quantify the relative importance of a term over time by
#            separating its growth from the overall growth in publications.
#
# Tasks:
#   1. Load the 3-Gram data file (trajetory_3_gram_1950_2010.parquet).
#   2. Select the term "messenger rna expression" and load the Total
#      Documents Published Annually from the external normalization file
#      (using the S1 search reference).
#   3. Create two separate time series plots (1950-2010) for this term:
#      a. Plot A: Absolute Frequency (raw count per year).
#      b. Plot B: Normalized Frequency (Term Count / Total Documents).
#   4. Explain how the term's trajectory differs between Plot A and Plot B.
#      Discuss why Normalized Frequency is the essential input for all
#      subsequent statistical analyses aimed at forecasting the term's
#      future trajectory.
# ==============================================================================

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION (User-Adjustable Parameters) ---

# File paths
TRAJECTORY_FILE = 'trajetory_3_gram_1950_2010.parquet'
NORMALIZATION_FILE = 'data/search_publication.xlsx'
NORMALIZATION_SEARCH_REF = 'S1'  # Broadest search to normalize against

# Term of interest (a 3-Gram, hence it must be in the 3-Gram Parquet file)
TERM = "messenger rna expression"

# Year ranges for the plots
PLOT_A_YEAR_RANGE = (1950, 2010)
PLOT_B_YEAR_RANGE = (1950, 2010)


# ==============================================================================
# TASK 1: Load the 3-Gram data file
# ==============================================================================

print("=" * 70)
print("TASK 1: Loading 3-Gram trajectory data")
print("=" * 70)

try:
    df_3gram = pd.read_parquet(TRAJECTORY_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"ERROR: Parquet file not found at '{TRAJECTORY_FILE}'. "
        "Ensure the data preparation pipeline has been run first."
    )

# --- Data Inspection ---
# Verify the structure and dimensions of the loaded Parquet file.
# This check ensures the file contains the expected columns (Term, Count, Year)
# and gives a quick overview of the data volume.

print(f"\n--- Data Inspection: {TRAJECTORY_FILE} ---")
print(f"Dimensions: {df_3gram.shape[0]:,} rows x {df_3gram.shape[1]} columns")
print(f"\nColumn types:\n{df_3gram.dtypes}")
print(f"\nFirst 5 rows:\n{df_3gram.head()}")

# --- Top-10 Most Frequent Terms ---
# Ranking the terms by total absolute frequency across the entire period
# helps confirm that the dataset is loaded correctly and provides context
# for interpreting the term of interest against the most dominant terms.

term_ranking = (
    df_3gram.groupby('Term')['Count']
    .sum()
    .reset_index(name='Total_Absolute_Frequency')
    .sort_values(by='Total_Absolute_Frequency', ascending=False)
)

print(f"\n--- Top 10 3-Gram Terms by Absolute Frequency (1950-2010) ---")
print(term_ranking.head(10).to_string(index=False))
print()


# ==============================================================================
# TASK 2: Select the term and load the normalization base
# ==============================================================================

print("=" * 70)
print("TASK 2: Selecting term and loading normalization base (S1)")
print("=" * 70)

# --- Load Total Publications from the external normalization file ---
try:
    total_docs_raw = pd.read_excel(NORMALIZATION_FILE)
except FileNotFoundError:
    raise FileNotFoundError(
        f"ERROR: Normalization file not found at '{NORMALIZATION_FILE}'."
    )

# Filter for the S1 search reference (broadest scope)
total_docs = total_docs_raw[
    total_docs_raw['Search ref'] == NORMALIZATION_SEARCH_REF
].copy()

# Rename columns and set Year as the index for merging
total_docs = total_docs.rename(columns={
    'Category': 'Year',
    'Publications (total)': 'Total_Documents'
})
total_docs['Year'] = total_docs['Year'].astype(int)
total_docs = total_docs.set_index('Year')[['Total_Documents']]

# --- Select the term of interest ---
term_data = df_3gram[df_3gram['Term'] == TERM].copy()

if term_data.empty:
    print(f"\nWARNING: The term '{TERM}' was not found in the 3-Gram database.")
    plt.close('all')
    exit()

term_data.set_index('Year', inplace=True)

# Group by year to get the Absolute Frequency (sum of counts per year)
term_data_grouped = term_data.groupby('Year')['Count'].sum().to_frame(
    name='Absolute_Frequency'
)

# Merge term counts with total document counts for normalization
term_data_normalized = term_data_grouped.merge(
    total_docs, left_index=True, right_index=True, how='left'
)

# Calculate Normalized Frequency per 1,000 documents
term_data_normalized['Normalized_Frequency'] = (
    term_data_normalized['Absolute_Frequency']
    / term_data_normalized['Total_Documents']
) * 1000

# --- Data Integrity Check ---
# Verify that Total_Documents shows expected growth over time.
# If the normalization base is flat or decreasing, the normalization
# will not correctly separate term growth from domain growth.

print(f"\n--- Data Integrity Check: S1 Total Publications ---")
print(f"Earliest years:\n{term_data_normalized.head(3)}")
print(f"\nLatest years:\n{term_data_normalized.tail(3)}")
print(f"\nTerm '{TERM}' successfully loaded with "
      f"{len(term_data_normalized)} yearly observations.")


# ==============================================================================
# TASK 3a: Plot A — Absolute Frequency (raw count per year)
# ==============================================================================

print("\n" + "=" * 70)
print("TASK 3a: Generating Plot A — Absolute Frequency")
print("=" * 70)

start_A, end_A = PLOT_A_YEAR_RANGE
plot_data_A = term_data_normalized.loc[start_A:end_A, 'Absolute_Frequency']

plt.figure(figsize=(10, 4))
plt.plot(plot_data_A.index, plot_data_A.values, label='Absolute Frequency')
plt.title(f'Plot A: Absolute Frequency of "{TERM}" ({start_A}-{end_A})')
plt.xlabel('Year')
plt.ylabel('Raw Count')
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================================================================
# TASK 3b: Plot B — Normalized Frequency (Term Count / Total Documents)
# ==============================================================================

print("=" * 70)
print("TASK 3b: Generating Plot B — Normalized Frequency")
print("=" * 70)

start_B, end_B = PLOT_B_YEAR_RANGE
plot_data_B = term_data_normalized.loc[start_B:end_B, 'Normalized_Frequency']

plt.figure(figsize=(10, 4))
plt.plot(plot_data_B.index, plot_data_B.values,
         label='Normalized Frequency', color='red')
plt.title(f'Plot B: Normalized Frequency of "{TERM}" ({start_B}-{end_B})')
plt.xlabel('Year')
plt.ylabel(
    f'Count per 1,000 Documents (Normalized to {NORMALIZATION_SEARCH_REF} Search)'
)
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================================================================
# TASK 4: Discussion — Absolute vs. Normalized Frequency
# ==============================================================================
#
# After running this script, compare Plot A and Plot B:
#
# - Plot A (Absolute Frequency) shows the raw count of the term per year.
#   This curve tends to rise simply because the total volume of scientific
#   publications grows exponentially over time. A term can appear to be
#   "emerging" when in reality it is merely keeping pace with the overall
#   growth of the domain.
#
# - Plot B (Normalized Frequency) divides the term count by the total
#   number of documents published in that year (S1 search). This isolates
#   the term's *relative importance* and reveals whether it is genuinely
#   gaining traction or simply riding the wave of increased publication
#   volume.
#
# Conclusion: Normalized Frequency is the essential input for all
# subsequent statistical analyses (Burst Detection, Growth Rate
# calculations, K-means clustering on temporal trajectories) because
# it removes the confounding effect of overall domain growth and
# provides a true measure of a term's trajectory.
# ==============================================================================

print("\n" + "=" * 70)
print("Exercise 5.1 completed successfully.")
print("Compare Plot A and Plot B to understand the difference between")
print("absolute and normalized frequency trajectories.")
print("=" * 70)