import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
import os
import json
import nltk
from nltk.tokenize import word_tokenize
import logging
import multiprocessing

# Logging Configuration
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- CONFIGURATION ---
# 1. Path to your Master Corpus of Documents (ORIGINAL JSON)
MASTER_CORPUS_FILE = 'data/advanced/documentos_com_full_text.json' 

# 2. Output paths for the models
OUTPUT_DIR = 'data/advanced/word2vec_temporal'
MODEL_EARLY_PATH = os.path.join(OUTPUT_DIR, 'word2vec_model_early.kv')
MODEL_LATE_PATH = os.path.join(OUTPUT_DIR, 'word2vec_model_late.kv')

# 3. Temporal and Model Parameters
SPLIT_YEAR = 2007  
VECTOR_SIZE = 100 
WINDOW = 5        
MIN_COUNT = 5     

# Stable Parameters (from your successful model)
WORKERS = 3         
ALGORITHM = 1       # 1 = Skip-gram (sg=1)

# Download required NLTK component if not present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


# --- STEP 1: LOAD, SEGMENT AND PREPARE THE CORPUS IN MEMORY ---

def load_and_segment_corpus(file_path, split_year):
    """
    Loads the JSON file, tokenizes and splits it into two corpora (Early/Late)
    based on publication year, using the correct keys from the JSON.
    """
    corpus_early_sentences = []
    corpus_late_sentences = []
    
    print(f"Step 1: Loading master corpus from: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"FATAL ERROR: JSON master corpus file not found at {file_path}.")

    print(f"Loaded {len(data)} documents for processing.")
    
    fail_count = 0 
    
    for doc in data:
        # WARNING: Using keys confirmed by the user
        full_text = doc.get('document_full_text')
        year_raw = doc.get('publication_year') # CORRECT KEY

        # Try converting the year to an integer, handling strings or nulls
        year = None
        try:
            if year_raw is not None:
                year = int(year_raw)
        except (ValueError, TypeError):
            pass # Ignore documents without a valid year

        # 1. Data Validation: Must have text and a valid year
        if isinstance(full_text, str) and full_text and isinstance(year, int):
            
            # Tokenization (same as your successful script)
            tokens = word_tokenize(full_text)
            cleaned_tokens = [w for w in tokens if w.isalpha() and len(w) > 1]
            
            if cleaned_tokens:
                # 2. Segmentation
                if year < split_year:
                    corpus_early_sentences.append(cleaned_tokens)
                else:
                    corpus_late_sentences.append(cleaned_tokens)
            else:
                fail_count += 1
        else:
            fail_count += 1

    if fail_count > 0:
        print(f"WARNING: {fail_count} documents were discarded due to missing text or invalid year.")
        
    if not corpus_early_sentences and not corpus_late_sentences:
         raise ValueError("Corpus preparation resulted in empty sentence lists. Check JSON keys ('document_full_text' and 'publication_year').")

    return corpus_early_sentences, corpus_late_sentences

# Execute preparation
corpus_early, corpus_late = load_and_segment_corpus(MASTER_CORPUS_FILE, SPLIT_YEAR)

print(f"EARLY Corpus (before {SPLIT_YEAR}): {len(corpus_early)} documents.")
print(f"LATE Corpus (from {SPLIT_YEAR} onwards): {len(corpus_late)} documents.")


# --- STEP 2: MODEL TRAINING ---
# Keep this block the same as the previous one, as parameters are stable.

os.makedirs(OUTPUT_DIR, exist_ok=True)
model_params = {
    'vector_size': VECTOR_SIZE,
    'window': WINDOW,
    'min_count': MIN_COUNT,
    'workers': WORKERS,
    'sg': ALGORITHM, 
    'epochs': 10 # Added epochs for consistency
}

# 2.1 Training the EARLY Model (W2VEarly)
print(f"\nStep 2.1: Starting EARLY model training (Skip-gram) with {WORKERS} worker(s)...")
model_early = Word2Vec(sentences=corpus_early, **model_params)
model_early.wv.save(MODEL_EARLY_PATH)
print(f"W2V_Early model trained and saved to: {MODEL_EARLY_PATH}")


# 2.2 Training the LATE Model (W2VLate)
print(f"\nStep 2.2: Starting LATE model training (Skip-gram) with {WORKERS} worker(s)...")
model_late = Word2Vec(sentences=corpus_late, **model_params)
model_late.wv.save(MODEL_LATE_PATH)
print(f"W2V_Late model trained and saved to: {MODEL_LATE_PATH}")


print("\n--- TRAINING COMPLETED ---")
print("The temporal models are ready! You can now proceed to Script 4.")