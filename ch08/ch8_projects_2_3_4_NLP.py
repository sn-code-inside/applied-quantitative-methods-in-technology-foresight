# ================================================================
# Chapter 8.2 — Deep Learning Applications in Foresight 
# ================================================================
# Dataset License: CORD-19 (CC BY 4.0)
# ------------------------------------------------
# This file automatically downloads an open dataset (CORD-19)
# and demonstrates text-based foresight analytics projects given in Section 8.2
# ================================================================
# -------- REQUIRED PACKAGES ------------------------------------
# Run these lines once if needed:
#pip install -U pandas numpy scikit-learn matplotlib tqdm sentence-transformers 
#pip install umap-learn 
#pip install -U tensorflow keras transformers networkx requests
#pip install -U tf-keras

import zipfile, os, io, re, json, requests
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    keras = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
try:
    from transformers import pipeline
except Exception:
    pipeline = None
try:
    import networkx as nx
except Exception:
    nx = None

# ================================================================
# DEVICE CONFIGURATION
# ------------------------------------------------
# This block automatically selects GPU if available, otherwise CPU.
# ================================================================
import torch
print("Torch CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
if torch.cuda.is_available():
    device_local = "cuda"
#elif torch.backends.mps.is_available():
#    device_local = "mps"  # for Apple Silicon
else:
    device_local = "cpu"
print(f"Using device: {device_local.upper()}")

def load_cord19_from_url(url="https://ai2-semanticscholar-cord-19.s3.amazonaws.com/latest/metadata.csv",
                         sample_size=10000):
    # ================================================================
    # Function: load_cord19_from_url
    # ------------------------------------------------
    # This function fetches an open CC-BY dataset directly from
    # the Semantic Scholar CORD-19 repository for reproducible analysis.
    # ================================================================
    print("Downloading sample from CORD-19 dataset...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), low_memory=False, nrows=sample_size)
    df = df.rename(columns=str.lower)
    df = df.loc[df['abstract'].notna(), ['title','abstract','publish_time']]
    df['year'] = pd.to_datetime(df['publish_time'], errors='coerce').dt.year.fillna(2020).astype(int)
    # --- Data cleaning: remove missing or empty values ---
    df = df[['title','abstract','year']].dropna(subset=['title', 'abstract'])
    df = df[df['abstract'].str.strip() != ""]
    df = df[df['title'].str.strip() != ""]
    print(f"Loaded {len(df)} records from CORD-19")
    return df[['title','abstract','year']]

def load_cord19_from_zip(zip_path="cord_19_dataset.zip", inner_name="cord19_df.csv"):
    print(f"Reading {inner_name} from {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(inner_name) as f:
            df = pd.read_csv(f, low_memory=False)
    df = df.rename(columns=str.lower)
    if 'year' not in df.columns:
        if 'publish_time' in df.columns:
            df['year'] = pd.to_datetime(df['publish_time'], errors='coerce').dt.year
        else:
            df['year'] = 2020
    df['year'] = df['year'].fillna(2020).astype(int)
    # --- Data cleaning: remove missing or empty values ---
    df = df[['title','abstract','year']].dropna(subset=['title', 'abstract'])
    df = df[df['abstract'].str.strip() != ""]
    df = df[df['title'].str.strip() != ""]
    print(f"Loaded {len(df)} rows.")
    return df[['title','abstract','year']]
# ================================================================
# PROJECT 2 — Emerging Technology Detection
# ------------------------------------------------
# Objective: Identify latent research clusters that indicate emerging
# technologies using Sentence-BERT embeddings + clustering.
# ================================================================
def project2_embeddings_clustering(df, n_clusters=8):
    """
    Identify emerging research clusters using Sentence-BERT embeddings + KMeans.
    Falls back to TF-IDF embeddings if SentenceTransformer cannot be loaded.
    """
    print("Generating embeddings and clustering (robust)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device_local)
        embeddings = model.encode(df["abstract"].tolist(), show_progress_bar=True)
    except Exception as e:
        print("Sentence-BERT not available or failed to load:", e)
        print("Falling back to TF-IDF vectorization.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=768, stop_words="english")
        embeddings = vec.fit_transform(df["abstract"].tolist()).toarray()
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    df["cluster"] = km.labels_
    freq = df.groupby(["year","cluster"]).size().reset_index(name="count")
    pivot = freq.pivot(index="year", columns="cluster", values="count").fillna(0)
    pivot.plot(figsize=(10,6), title="Cluster Frequency by Year (Emerging Topics)")
    plt.tight_layout(); 
    OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_ch8_project_2")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "cluster_trends.png"), dpi=600)
    print("Clustering complete and plot saved as cluster_trends.png")
    # --- Display sample titles per cluster ---
    for c in sorted(df["cluster"].unique()):
        print(f"\nCluster {c} sample titles:")
        print(df.loc[df["cluster"]==c, "title"].head(3).to_string(index=False))
    
    # --- Show top TF-IDF keywords per cluster ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", max_features=10)
    for c in sorted(df["cluster"].unique()):
        subset = df.loc[df["cluster"] == c, "abstract"]
        subset = subset.dropna().astype(str)
        subset = subset[subset.str.len() > 20]  # very short abstracts ignored
        if len(subset) > 5:
            try:
                X = vec.fit_transform(subset)
                top_terms = vec.get_feature_names_out()
                print(f"Cluster {c} keywords: {', '.join(top_terms)}")
            except ValueError:
                print(f"Cluster {c}: empty vocabulary (too short or stopwords only). Skipped.")
    return df, pivot


# ================================================================
# PROJECT 3 — Trend Forecasting with LSTM
# ------------------------------------------------
# Objective: Predict topic growth using a recurrent neural network (LSTM).
# ================================================================
def project3_trend_forecasting(freq_pivot):
    if keras is None:
        raise ImportError("TensorFlow/Keras not available.")
    # Force TensorFlow to use GPU if available, otherwise CPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print("TensorFlow using GPU:", physical_devices)
    else:
        print("TensorFlow running on CPU.")
    # Normalize topic frequencies between 0 and 1 to stabilize loss
    freq_pivot = freq_pivot / freq_pivot.max().max()

    X, y = [], []
    window = 3
    for c in freq_pivot.columns:
        series = freq_pivot[c].values.astype("float32")
        for t in range(len(series)-window):
            X.append(series[t:t+window])
            y.append(series[t+window])
    X = np.array(X)[:, :, None]; y = np.array(y)
    model = keras.Sequential([layers.LSTM(32, input_shape=(window,1)), layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    hist = model.fit(X, y, epochs=30, batch_size=8, verbose=0, validation_split=0.2)
    plt.plot(hist.history["loss"], label="train"); plt.plot(hist.history["val_loss"], label="val")
    plt.legend();plt.title("LSTM Training Loss (Normalized Topic Trends)")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.tight_layout(); 
    OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_ch8_project_3")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "lstm_loss.png"), dpi=600)
    return model

# ================================================================
# PROJECT 4 — Entity-Based Strategic Mapping
# ------------------------------------------------
# Objective: Identify key organizations/countries leading research using NER.
# ================================================================
def project4_ner_network(df):
    try:
        from transformers import pipeline
    except Exception as e:
        print("Transformers pipeline could not be imported.")
        print("Project 3 — Entity-Based Strategic Mapping will be skipped.")
        print("Import error:", e)
        return pd.DataFrame(columns=["source", "target", "weight"])
    try:
        nlp = pipeline(
            "ner",
            aggregation_strategy="simple",
            model="dslim/bert-base-NER"
        )
        entities = []
        for text in tqdm(df["abstract"].head(200).tolist(), desc="NER"):
            try:
                ents = nlp(str(text)[:500])
                entities.append([
                    e["word"] for e in ents
                    if e.get("entity_group") in ["ORG", "LOC"]
                ])
            except Exception:
                entities.append([])
        pairs = {}
        for ents in entities:
            ents = sorted(set(ents))
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    key = (ents[i], ents[j])
                    pairs[key] = pairs.get(key, 0) + 1
        edges = pd.DataFrame([
            {"source": a, "target": b, "weight": w}
            for (a, b), w in pairs.items()
        ])
        OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_ch8_project_4")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        edges.to_csv(os.path.join(OUTPUT_DIR, "entity_network.csv"), index=False)
        print("Project 3 completed successfully.")
        print("Output saved as entity_network.csv.")
        return edges
    except Exception as e:
        print("Project 3 failed during execution.")
        print("Error:", e)
        return pd.DataFrame(columns=["source", "target", "weight"])
# ================================================================
# Case Study — Sentiment-Driven Policy Insight
# ------------------------------------------------
# Objective: Evaluate scientific tone to reveal policy optimism/pessimism trends.
# ================================================================
def case_study_sentiment(df):
    clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    sub = df[df["abstract"].str.contains("policy|vaccine|regulation", case=False, na=False)]
    labels = [clf(t[:400])[0]["label"] for t in tqdm(sub["abstract"].tolist(), desc="Sentiment")]
    sub["sentiment"] = labels
    yearly = sub.groupby(["year","sentiment"]).size().reset_index(name="count")
    OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_ch8_case_study")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yearly.to_csv(os.path.join(OUTPUT_DIR, "sentiment_by_year.csv"), index=False)
    return yearly

# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------
    # Use this option to download the CORD-19 dataset directly.
    # df = load_cord19_from_url()

    # If you already have the local ZIP file, use the following line instead:
    df = load_cord19_from_zip()

    print(df.head())

    # ------------------------------------------------------------
    # Project 1 — Emerging Technology Detection
    # ------------------------------------------------------------
    df, freq = project2_embeddings_clustering(df)

    dominant_years = freq.idxmax(axis=0)
    print("\nDominant publication year per cluster:")
    print(dominant_years)

    print("\nProject 1 of Chapter 8 Section 2 has been completed.")

    # ------------------------------------------------------------
    # Project 2 — Trend Forecasting with LSTM
    # ------------------------------------------------------------
    model = project3_trend_forecasting(freq)

    print("\nProject 2 of Chapter 8 Section 2 has been completed.")

    # ------------------------------------------------------------
    # Project 3 — Entity-Based Strategic Mapping
    # ------------------------------------------------------------
    edges = project4_ner_network(df)
    
    print("\nProject 3 of Chapter 8 Section 2 has been completed.")

    # ------------------------------------------------------------
    # Case Study — Sentiment-Driven Policy Insight
    # ------------------------------------------------------------
    yearly_sent = case_study_sentiment(df)

    print("\nCase Study of Chapter 8 Section 2 has been completed.")

    # ------------------------------------------------------------
    # Final Message
    # ------------------------------------------------------------
    print("\n Done ")
   