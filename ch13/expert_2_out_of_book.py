# Import libraries
import pandas as pd
import numpy as np
import json
import os

# --- FILE AND PARAMETER CONFIGURATION ---
NORMALIZATION_FILE = 'data/search_publication.xlsx' 
CANDIDATES_FILE_XLSX = 'data/expert/Final_Propostas_sinais_Fracos.xlsx'
OUTPUT_FILE_JSON = 'data/expert/weak_signal_candidates_structured.json'

NORMALIZATION_SEARCH_REF = 'S1' # Normalization filter

# CRITICAL FILE LIST: Include all necessary Parquet files to find your N-grams
TRAJECTORY_FILES = [
    'trajetory_1_gram_1950_2010.parquet',
    'trajetory_2_gram_1950_2010.parquet',
    'trajetory_3_gram_1950_2010.parquet',
    'trajetory_4_gram_1950_2010.parquet',
    'trajetory_5_gram_1950_2010.parquet',
    'trajetory_6_gram_1950_2010.parquet',
    'trajetory_7_gram_1950_2010.parquet'
]

# Define period for Burst Score calculation
BURST_PERIOD_START = 2007
BURST_PERIOD_END = 2010
YEARS_FOR_SCORE = list(range(BURST_PERIOD_START, BURST_PERIOD_END + 1))


# --- STEP 1: LOAD FREQUENCY AND NORMALIZATION DATA ---

print("Step 1: Loading, consolidating and normalizing all trajectories...")

# Load normalization base (Total Documents)
try:
    total_docs_raw = pd.read_excel(NORMALIZATION_FILE)
    total_docs = total_docs_raw[total_docs_raw['Search ref'] == NORMALIZATION_SEARCH_REF].copy()
    total_docs = total_docs.rename(columns={'Category': 'Year', 'Publications (total)': 'Total_Documents'})
    total_docs['Year'] = total_docs['Year'].astype(int)
    total_docs = total_docs.set_index('Year')[['Total_Documents']]
except FileNotFoundError as e:
    raise FileNotFoundError(f"ERROR: Normalization file not found: {e}")

# Consolidate all Parquet files
all_trajectories = []
for file in TRAJECTORY_FILES:
    try:
        df = pd.read_parquet(file)
        all_trajectories.append(df)
    except FileNotFoundError:
        print(f"WARNING: Trajectory file not found and skipped: {file}")

if not all_trajectories:
    raise Exception("FATAL ERROR: No Parquet trajectory base was loaded.")

df_trajectories = pd.concat(all_trajectories, ignore_index=True)

# Normalization: Merge raw count with total documents
df_trajectories = df_trajectories.merge(
    total_docs, left_on='Year', right_index=True, how='left'
)

# Normalized Frequency Calculation
df_trajectories['Normalized_Frequency'] = (
    df_trajectories['Count'] / df_trajectories['Total_Documents'].replace(0, np.nan)
) * 1000

print(f"All trajectory bases consolidated. Total lines: {len(df_trajectories)}")


# --- STEP 2: PIVOT DATA FOR GROWTH CALCULATION ---

# Pivot the table (Years as index, Terms as columns, Normalized_Frequency as value)
pivot_table = df_trajectories.pivot_table(
    index='Year', 
    columns='Term', 
    values='Normalized_Frequency', 
    fill_value=0
)

# 2.1 Calculate Percentage Growth (Year-over-Year)
# Replaces 0 with NaN to avoid infinity in division when previous frequency is zero.
growth_pct = pivot_table.pct_change().replace([np.inf, -np.inf], np.nan) 

print("\nStep 2: Percentage growth table (YoY) prepared.")


# --- STEP 3: LOAD CANDIDATES AND CALCULATE REAL BURST SCORE ---

print("\nStep 3: Loading the final list of candidates...")
try:
    # Load the N-grams list
    df_candidates_raw = pd.read_excel(CANDIDATES_FILE_XLSX, header=None, names=['ngram'], usecols=[0])
except FileNotFoundError:
    raise FileNotFoundError(f"ERROR: Candidates file not found at {CANDIDATES_FILE_XLSX}.")

# Cleaning and initial structure
df_candidates_raw['ngram'] = df_candidates_raw['ngram'].astype(str).str.strip()
df_final = df_candidates_raw.copy()
df_final['n_gram'] = df_final['ngram'].apply(lambda x: len(x.split()))


# 3.1 Isolating candidate term columns in the growth table
# Filter only terms present in our candidate list AND that exist in the trajectory base
valid_candidates = [term for term in df_final['ngram'].tolist() if term in growth_pct.columns]
df_final = df_final[df_final['ngram'].isin(valid_candidates)].reset_index(drop=True)

growth_candidates_data = growth_pct[valid_candidates]

# 3.2 Focus only on the Burst period (2007-2010)
growth_burst_period = growth_candidates_data.loc[YEARS_FOR_SCORE]

# 3.3 Calculate Real Burst Score (Average YoY Growth in the 2007-2010 period)
# Use the average, ignoring NaN (for years when frequency was zero, resulting in undefined growth)
burst_scores_series = growth_burst_period.mean(skipna=True)

# 3.4 Map scores to the final table
df_final['burst_score'] = df_final['ngram'].map(burst_scores_series) * 100 # Multiply by 100 to get %

print(f"Real Burst Score calculation completed. {len(df_final)} validated and ready candidates.")


# --- STEP 4: SAVE IN STRUCTURED JSON ---

# Convert DataFrame to a list of dictionaries
candidates_list = df_final.to_dict('records')

# Ensure the directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE_JSON), exist_ok=True)

with open(OUTPUT_FILE_JSON, 'w', encoding='utf-8') as f:
    json.dump(candidates_list, f, ensure_ascii=False, indent=4)

print(f"\nStructured data with Real Burst Score saved to: {OUTPUT_FILE_JSON}")