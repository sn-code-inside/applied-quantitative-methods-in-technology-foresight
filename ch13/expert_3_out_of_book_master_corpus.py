import pandas as pd
import json
import os
import re

# --- CONFIGURATION ---
# Input JSON file (with 'fulltext' and 'year')
INPUT_JSON_FILE = 'data/advanced/documentos_com_full_text.json' 
# Output Parquet file (the Master Corpus missing in Script 5)
# *** CORRECTED PATH ***
OUTPUT_PARQUET_FILE = 'data/expert/master_corpus_tokenized.parquet' 

# Columns to use
YEAR_COLUMN = 'publication_year'
TEXT_COLUMN = 'document_full_text'


# --- STEP 1: LOAD AND PREPARE DATA ---

print(f"Step 1: Loading data from JSON at {INPUT_JSON_FILE}...")

# 1.1 Load the JSON
try:
    # Read JSON line by line
    with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
except FileNotFoundError:
    raise FileNotFoundError(f"FATAL ERROR: Input JSON file not found at {INPUT_JSON_FILE}.")

# 1.2 Select and Rename columns
df = df[[YEAR_COLUMN, TEXT_COLUMN]].copy()
df.rename(columns={
    YEAR_COLUMN: 'Year',
    TEXT_COLUMN: 'Processed_Text'
}, inplace=True)

# 1.3 Clean null rows and ensure types
df = df.dropna(subset=['Year', 'Processed_Text'])
df['Year'] = df['Year'].astype(int)

print(f"Total valid documents loaded: {len(df)}")


# --- STEP 2: CONVERT TEXT TO LIST OF TOKENS ---
# Gensim Word2Vec requires a LIST of words for each document.

def string_to_token_list(text):
    """Converts the cleaned text string into a list of tokens/words."""
    if isinstance(text, str):
        # Just splits the string into words (tokens)
        return text.split() 
    return []

print("Step 2: Converting text from string to list of tokens...")

# Apply conversion
df['Processed_Text'] = df['Processed_Text'].apply(string_to_token_list)

# Filter any rows where the processed text resulted in an empty list
df = df[df['Processed_Text'].apply(len) > 0].reset_index(drop=True)

print(f"Final total of documents for training: {len(df)}")


# --- STEP 3: SAVE THE MASTER CORPUS IN PARQUET FORMAT ---

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_PARQUET_FILE), exist_ok=True)

# Save the DataFrame in Parquet format
df.to_parquet(OUTPUT_PARQUET_FILE, index=False)

print(f"\n--- GENERATION COMPLETED ---")
print(f"Master Corpus Tokenized successfully saved at: {OUTPUT_PARQUET_FILE}")