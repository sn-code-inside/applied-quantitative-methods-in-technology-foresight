
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import json
import os 
import glob 

# Define the folder path and the N-gram sizes required for ALL exercises
folder_path = 'data/triennial_data_folder/'
# NGRAM_SIZES includes 3, which is required for the term "messenger rna expression"
NGRAM_SIZES = [1, 2, 3, 4, 5, 6,7] 

# Initialize dictionaries to store data for each N-gram size
data_ngrams = {n: [] for n in NGRAM_SIZES}
total_docs_data = {} # Dictionary to store {Year: Total_Count_All_Ngrams}

# Use glob to safely find all JSON files in the folder
json_files = glob.glob(os.path.join(folder_path, '*.json'))

if not json_files:
    raise ValueError(f"ERROR: No JSON files found in {os.path.abspath(folder_path)}. Check the path.")

print(f"Found {len(json_files)} JSON files. Starting consolidation for {NGRAM_SIZES} N-grams...")

# Iterate over all found file paths (each file covers multiple years)
for file_path in json_files:
    
    # Load the JSON file content {Year: {term: count}}
    with open(file_path, 'r') as f:
        triennial_period_data = json.load(f)
        
    # Iterate through each year (key) inside the JSON file (e.g., '1950', '1951')
    for year_str, term_counts in triennial_period_data.items():
        year = int(year_str)
        
        # --- CALCULATE TOTAL DOCUMENTS (Normalization Base) ---
        total_docs_data[year] = sum(term_counts.values())
        
        # Convert the terms/counts dictionary into a temporary DataFrame
        temp_df = pd.DataFrame(list(term_counts.items()), columns=['Term', 'Count'])
        temp_df['Year'] = year 
        
        # Infer N-gram size
        temp_df['N_gram_Size'] = temp_df['Term'].apply(lambda x: len(x.split()))
        
        # Filter and append data for each required N-gram size
        for n in NGRAM_SIZES:
            df_filtered = temp_df[temp_df['N_gram_Size'] == n].copy()
            if not df_filtered.empty:
                data_ngrams[n].append(df_filtered)

# --- SAVE ALL N-GRAM TRAJECTORIES AND TOTAL DOCUMENTS IN PARQUET FORMAT ---

# Consolidate and save each N-gram DataFrame
for n in NGRAM_SIZES:
    if data_ngrams[n]:
        df_n_gram = pd.concat(data_ngrams[n], ignore_index=True)
        output_filename = f'trajetory_{n}_gram_1950_2010.parquet'
        # Using Parquet for efficiency and large data volumes
        df_n_gram.to_parquet(output_filename, index=False)
        print(f"Intermediate {n}-Gram trajectory saved to: {output_filename}")
    else:
        print(f"WARNING: No {n}-Gram data found.")

# Prepare Total Documents for merging and save it (using Parquet as well)
total_docs = pd.Series(total_docs_data).to_frame(name='Total_Documents')
total_docs.index.name = 'Year'
total_docs_filename = 'total_documents_annual_1950_2010.parquet'
total_docs.to_parquet(total_docs_filename)
print(f"Total Documents saved to: {total_docs_filename}")
