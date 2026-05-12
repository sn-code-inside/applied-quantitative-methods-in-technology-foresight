# 6 Word Embedding Models for Weak Signal Detection: From Corpus Preparation to Semantic Vigilance (Tutorial)
# Import necessary libraries
import os
import json
from gensim.models import Word2Vec, KeyedVectors # KeyedVectors added for clarity on loading/saving
import nltk
from nltk.tokenize import word_tokenize

# --- CONFIGURATION ---
# Define the file path for the consolidated raw data
FILE_PATH = 'data/advanced/documentos_com_full_text.json'
MODEL_DIR = 'data/advanced'
MODEL_PATH = os.path.join(MODEL_DIR, 'word2vec_model_100d.kv')

# Ensure the output directory exists for saving the model
os.makedirs(MODEL_DIR, exist_ok=True) 

# Download necessary NLTK components (run once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- STEP 1: LOAD FULL TEXT AND PREPARE CORPUS FOR WORD2VEC ---

def prepare_corpus(file_path):
    """
    Loads text data from the JSON file and tokenizes the 'document_full_text' field 
    into a list of lists of words (corpus format required by Word2Vec), handling non-string values.
    """
    corpus_sentences = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"ERROR: Raw text file not found at {file_path}.")

    print(f"Loaded {len(data)} documents for processing.")

    for doc in data:
        full_text = doc.get('document_full_text')
        
        # CRITICAL FIX: Ensure the full_text is a non-empty string before processing
        if isinstance(full_text, str) and full_text:
            # Tokenize the entire text into words
            tokens = word_tokenize(full_text) # Assumes the text is already lowercased/cleaned as per data prep
            
            # Final Cleaning: Keep only alphabetic tokens longer than one character
            cleaned_tokens = [w for w in tokens if w.isalpha() and len(w) > 1]
            
            if cleaned_tokens:
                corpus_sentences.append(cleaned_tokens)
                
    if not corpus_sentences:
        raise ValueError("Corpus preparation resulted in an empty sentence list. Check data integrity.")
        
    return corpus_sentences

# Generate the corpus
corpus = prepare_corpus(FILE_PATH)
print(f"Total tokenized items (documents/sentences) prepared: {len(corpus)}")
print("Sample tokens from a document:", corpus[0][:15])


# --- STEP 2: TRAINING AND SAVING THE WORD2VEC MODEL ---

print("\n--- STEP 2: TRAINING AND SAVING THE WORD2VEC MODEL ---")
print("Starting Word2Vec Model Training...")

# Initialize and train the model in one optimized step
model = Word2Vec(
    sentences=corpus,
    vector_size=100,      
    window=5,             
    min_count=5,          
    sg=1,                 
    epochs=10,            
    workers=3             
)

print(f"Word2Vec Model Training Complete. Vocabulary Size: {len(model.wv.index_to_key)}")

# --- MODEL PERSISTENCE ---
try:
    # Save KeyedVectors to disk for quick loading in future sessions
    model.wv.save(MODEL_PATH)
    print(f"\nSUCCESS: Word2Vec model (KeyedVectors) saved to disk at: {MODEL_PATH}")
except Exception as e:
    print(f"\nERROR: Could not save model to disk at {MODEL_PATH}. Reason: {e}")


# --- STEP 3: SEMANTIC VALIDATION AND VIGILANCE ---

# --- TASK 3a: VALIDATION AND NOISE FILTERING ---

test_concept = 'mrna' 
test_debris = '000011' 

def validate_token(token, model):
    """Checks if a token has a meaningful vector representation and displays neighbors."""
    if token in model.wv:
        print(f"\nVALID: Token '{token}' found in the model's vocabulary.")
        
        print(f"Nearest semantic neighbors to '{token}':")
        try:
             # Most_similar calculates cosine similarity
             for word, score in model.wv.most_similar(token, topn=5):
                print(f"  - {word}: {score:.4f}")
        except KeyError:
             print(f"  - Not enough context to find neighbors.")
    else:
        # Automated filter for low-frequency noise (min_count threshold)
        print(f"\nDEBRIS: Token '{token}' not found in vocabulary (Filtered by min_count).")

print("\n--- TASK 3a: SEMANTIC VALIDATION (Noise Filtering) ---")
# Run validation checks
validate_token(test_concept, model)
validate_token(test_debris, model)

# --- TASK 3b: SEMANTIC VIGILANCE ---

core_concept = 'protein' 

print(f"\n--- TASK 3b: SEMANTIC VIGILANCE (Contextual Monitoring for '{core_concept}') ---")

if core_concept in model.wv:
    # Identify the top 10 concepts most similar in the vector space
    top_neighbors = model.wv.most_similar(core_concept, topn=10)
    print(f"Top 10 Semantically Closest Concepts to '{core_concept}':")
    for word, score in top_neighbors:
        print(f"  - {word}: {score:.4f}")
else:
    print(f"Core concept '{core_concept}' not frequent enough to be in the model.")