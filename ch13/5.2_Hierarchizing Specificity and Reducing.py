# Exercise 5.2: Hierarchizing Specificity and Reducing Noise via N-Gram Length

# Import libraries
import pandas as pd
import numpy as np

# Define file paths (Loading Parquet files)
path_2g = 'trajetory_2_gram_1950_2010.parquet'
path_6g = 'trajetory_6_gram_1950_2010.parquet'
min_abs_freq = 10
specificity_percentage = 0.05 # Top 5%

# --- STEP 1: LOAD FILTERED N-GRAM DATA AND CALCULATE TOTAL ABSOLUTE FREQUENCY ---

# Load 2-Gram and 6-Gram data directly from the generated Parquet files
df_2g_raw = pd.read_parquet(path_2g)
df_6g_raw = pd.read_parquet(path_6g)

# Calculate total absolute frequency per term across the entire period
df_2g_total = df_2g_raw.groupby('Term')['Count'].sum().reset_index(name='Absolute_Frequency')
df_6g_total = df_6g_raw.groupby('Term')['Count'].sum().reset_index(name='Absolute_Frequency')

# --- STEP 2: APPLY ABSOLUTE FREQUENCY FILTER (NOISE REDUCTION) ---

# Apply the filter: Remove all terms with an Absolute Frequency below 10
df_2g_filtered = df_2g_total[df_2g_total['Absolute_Frequency'] >= min_abs_freq].copy()
df_6g_filtered = df_6g_total[df_6g_total['Absolute_Frequency'] >= min_abs_freq].copy()

# --- STEP 3: APPLY SPECIFICITY CRITERION (TOP 5%) ---

# Sort by Absolute Frequency (proxy for total relevance)
total_norm_freq_2g = df_2g_filtered.sort_values(by='Absolute_Frequency', ascending=False)
total_norm_freq_6g = df_6g_filtered.sort_values(by='Absolute_Frequency', ascending=False)

# Select the top 5% of terms in each filtered list
top_5_percent_2g = total_norm_freq_2g.head(int(len(total_norm_freq_2g) * specificity_percentage))
top_5_percent_6g = total_norm_freq_6g.head(int(len(total_norm_freq_6g) * specificity_percentage))

# Display the top 50 terms for manual review and comparison
print("\n--- 5.2 Results ---")
print("Top 50 terms (2-Grams - Generic):\n", top_5_percent_2g.head(50))
print("\nTop 50 terms (6-Grams - Specific):\n", top_5_percent_6g.head(50))