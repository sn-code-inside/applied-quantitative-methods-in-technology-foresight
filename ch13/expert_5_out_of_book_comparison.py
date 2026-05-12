import pandas as pd
import numpy as np
import json
import os
from gensim.models import KeyedVectors

# --- FILE AND MODEL CONFIGURATION ---
MODEL_EARLY_PATH = os.path.join('data/advanced/word2vec_temporal', 'word2vec_model_early.kv') 
MODEL_LATE_PATH = os.path.join('data/advanced/word2vec_temporal', 'word2vec_model_late.kv') 
# Input file from Script 3
WEAK_SIGNALS_FILE = 'data/expert/weak_signal_candidates_structured.json'

# Using the output directory from Script 5 in the 'expert' folder
OUTPUT_DIR = 'data/expert/word2vec_temporal'
# Output file for the final ranking (Top 100)
RANKED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'ranked_signals_top100_displacement.json')


# --- STEP 1: LOAD DATA AND MODELS ---

print(f"Step 1: Loading candidates and temporal models...")

# 1.1 Load Structured Candidates
try:
    with open(WEAK_SIGNALS_FILE, 'r', encoding='utf-8') as f:
        candidates_list = json.load(f)
    weak_signals_df = pd.DataFrame(candidates_list)
except FileNotFoundError:
    raise FileNotFoundError(f"ERROR: Structured JSON file not found at {WEAK_SIGNALS_FILE}.")

# 1.2 Load Word2Vec Models
try:
    model_early = KeyedVectors.load(MODEL_EARLY_PATH)
    model_late = KeyedVectors.load(MODEL_LATE_PATH)
    print("Word2Vec Models (Early and Late) loaded successfully.")
except Exception as e:
    # This error is crucial: if models do not exist, Script 5 was not run or the path is wrong.
    print(f"\nFATAL ERROR: Failed to load Word2Vec models. Check if Script 5 was run and paths are correct.")
    print(f"Detail: {e}")
    raise


# --- STEP 2: VECTOR SUMMATION AND DISPLACEMENT CALCULATION FUNCTIONS ---

def get_ngram_vector(ngram, model, min_valid_words=0.5):
    """Calculates the vector of an N-gram by summing the vectors of its component words."""
    
    words = str(ngram).lower().split() 
    valid_vectors = []
    
    for word in words:
        if word in model.key_to_index:
            valid_vectors.append(model.get_vector(word))
    
    # Coherence Filter: Only N-grams with sufficient component words
    if len(valid_vectors) < len(words) * min_valid_words:
        return None
    
    if not valid_vectors:
        return None
    
    # Returns the sum of the vectors
    return np.sum(valid_vectors, axis=0)

def calculate_displacement(row):
    """Calculates the Cosine Displacement (Semantic Displacement) for an N-gram."""
    ngram = row['ngram']
    
    vector_early = get_ngram_vector(ngram, model_early)
    vector_late = get_ngram_vector(ngram, model_late)
    
    # Ignore if N-gram is not valid in one of the periods (lack of context)
    if vector_early is None or vector_late is None:
        return np.nan
    
    # Calculate similarity
    norm_early = np.linalg.norm(vector_early)
    norm_late = np.linalg.norm(vector_late)
    
    if norm_early == 0 or norm_late == 0:
        return np.nan
        
    similarity = np.dot(vector_early, vector_late) / (norm_early * norm_late)
    
    # Displacement = 1 - Similarity (Semantic Distance)
    return 1 - similarity

# --- STEP 3: EXECUTION AND FINAL RANKING (WITH PERSISTENCE) ---

print("\nStep 3: Calculating Semantic Displacement for the candidates...")

# Apply calculation
weak_signals_df['Semantic_Displacement'] = weak_signals_df.apply(calculate_displacement, axis=1)

# Ranking: The highest Displacement is the highest priority (Context Change)
ranked_candidates = weak_signals_df.sort_values(by='Semantic_Displacement', ascending=False, na_position='last')

# 3.1 Display Top 20 on console
print("\n--- Final Results: Ranking by Semantic Novelty ---")
print("The highest ranked candidates represent the greatest change in meaning.")

top_results_display = ranked_candidates[['ngram', 'n_gram', 'burst_score', 'Semantic_Displacement']].head(20).dropna()
print(top_results_display.to_markdown(index=False, floatfmt=".4f"))

# 3.2 Persistence: Save the Top 100 ranked candidates in JSON
top_100_ranked = ranked_candidates.head(100).dropna(subset=['Semantic_Displacement'])

try:
    # Creates the directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Converts the DataFrame to a list of dictionaries before saving
    top_100_ranked[['ngram', 'n_gram', 'burst_score', 'Semantic_Displacement']].to_json(
        RANKED_OUTPUT_FILE, 
        orient='records', 
        indent=4
    )
    print(f"\nSUCCESS: Top {len(top_100_ranked)} ranked candidates saved to: {RANKED_OUTPUT_FILE}")
except Exception as e:
    print(f"\nERROR: Failed to save the Top 100 ranking. Detail: {e}")


# 3.3 Analysis Conclusion
discarded_count = ranked_candidates['Semantic_Displacement'].isna().sum()
total_count = len(ranked_candidates)
print(f"\nFinal Note: {discarded_count} out of {total_count} candidates were discarded (not found or poorly represented in one of the periods).")