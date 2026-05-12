# Exercise 5.4: Identifying Technological Emergence (Burst Detection and Inflection Points)

# Import libraries
import pandas as pd
import numpy as np

# --- CONFIGURATION (Alignment with Ex. 5.1 Normalization) ---

path_4g = 'trajetory_4_gram_1950_2010.parquet'
NORMALIZATION_FILE = 'data/search_publication.xlsx' # External normalization file
NORMALIZATION_SEARCH_REF = 'S1' # Filter for the S1 search

# Define periods for analysis
PRIOR_PERIOD_END = 2006
RECENT_PERIOD_START = 2007

# --- STEP 1: LOAD DATA, NORMALIZE, AND PIVOT ---

# Load 4-Gram data
df_4g_raw = pd.read_parquet(path_4g)

# --- CORRECTION: Load and prepare the correct Total Documents base (S1 Search) ---
total_docs_raw = pd.read_excel(NORMALIZATION_FILE)
total_docs = total_docs_raw[total_docs_raw['Search ref'] == NORMALIZATION_SEARCH_REF].copy()
total_docs = total_docs.rename(columns={'Category': 'Year', 'Publications (total)': 'Total_Documents'})
total_docs = total_docs[['Year', 'Total_Documents']] # Keep only necessary columns

# Merge 4-Gram counts with the CORRECT Total Documents to calculate Normalized Frequency
df_4g = df_4g_raw.merge(total_docs, on='Year', how='left')
df_4g['Normalized_Frequency'] = (df_4g['Count'] / df_4g['Total_Documents']) * 1000 

# Pivot the table to prepare for year-over-year comparison
pivot_table = df_4g.pivot_table(index='Year', columns='Term', values='Normalized_Frequency', fill_value=0)

# --- STEP 2: APPLY EMERGENCE CRITERION ---

# Sum normalized frequency in the prior period (1950-2007)
prior_sum = pivot_table.loc[pivot_table.index <= PRIOR_PERIOD_END].sum()
# Sum normalized frequency in the recent period (2008-2010)
post_sum = pivot_table.loc[pivot_table.index >= RECENT_PERIOD_START].sum()

# Identify terms with prior sum == 0 AND post sum > 0
emergence_terms = (prior_sum == 0) & (post_sum > 0)
emergence_candidates = emergence_terms[emergence_terms].index.tolist()

print("\n--- 5.4 Results ---")
print("Emergence Candidates (Terms Born Recently - Top 10):\n", emergence_candidates[:10]) 

# --- STEP 3: APPLY SIGNIFICANT GROWTH CRITERION (BURST DETECTION) ---

df_growth = pivot_table.copy()

# Calculate Year-over-Year (YoY) growth from 2009 to 2010
growth_2010 = (df_growth.loc[2010] - df_growth.loc[2009]) / df_growth.loc[2009].replace(0, np.nan)

# Filter out the pure emergence candidates to focus on growth of established terms
growth_2010 = growth_2010[~growth_2010.index.isin(emergence_candidates)]

# Identify terms with >= 100% growth (the "Burst")
significant_growth = growth_2010[growth_2010 >= 1.0].sort_values(ascending=False)
top_10_growth = significant_growth.head(10)

print("\nTop 10 Significant Growth Candidates (Inflection Points):\n", top_10_growth)