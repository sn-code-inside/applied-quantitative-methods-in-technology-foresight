# Exercise 5.3: Conceptual Clustering and Thematic Grouping (TF-IDF and K-means)

# Import necessary libraries
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Define file path and load 5-Gram data
path_5g = 'trajetory_5_gram_1950_2010.parquet'
num_clusters = 5

# --- STEP 1: LOAD FILTERED 5-GRAM DATA AND CONVERT TO TF-IDF VECTORS ---

# Load 5-Gram data
df_5g_raw = pd.read_parquet(path_5g)

# Get the unique list of 5-Gram terms
terms = df_5g_raw['Term'].unique()

# Convert terms into TF-IDF vectors (terms are treated as 'documents')
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(terms)

# --- STEP 2: APPLY K-MEANS CLUSTERING ---

# Apply K-means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10) 
kmeans.fit(tfidf_matrix)

# Map terms back to their assigned cluster
df_results = pd.DataFrame({'Term': terms, 'Cluster': kmeans.labels_})

# Identify the Largest Cluster and its top terms
largest_cluster = df_results['Cluster'].value_counts().idxmax()
largest_cluster_terms = df_results[df_results['Cluster'] == largest_cluster].head(10)

print("\n--- 5.3 Results ---")
print("Top 10 Terms in Largest Cluster:\n", largest_cluster_terms)

# --- STEP 3: EVALUATE ISOLATED TERMS ---

# Identify terms in very small clusters (potential noise or thematic gaps)
cluster_sizes = df_results['Cluster'].value_counts()
small_clusters = cluster_sizes[cluster_sizes <= 2].index.tolist()
isolated_terms = df_results[df_results['Cluster'].isin(small_clusters)]

print("\nIsolated Terms (Potential Noise/Gaps):\n", isolated_terms)