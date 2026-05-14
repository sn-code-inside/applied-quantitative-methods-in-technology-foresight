"""
Topic Modeling and Text Analysis Module
Chapter 12: Data-Led Technology Roadmapping

This module provides functions for text preprocessing,
LDA topic modeling, and topic evolution analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Download NLTK data (run once)
import nltk
try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer


def preprocess_text(text: str, custom_stopwords: list = None) -> str:
    """
    Preprocess text for topic modeling.
    
    Steps: lowercase, remove special chars, tokenize,
           remove stopwords, lemmatize
           
    Parameters:
        text: Raw text string
        custom_stopwords: Additional domain-specific stopwords
        
    Returns:
        Preprocessed text string
    """
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    
    # Add custom stopwords for academic texts
    academic_stopwords = {'study', 'research', 'paper', 'propose', 'proposed',
                         'method', 'approach', 'result', 'results', 'using',
                         'based', 'show', 'shown', 'use', 'used', 'also'}
    stop_words.update(academic_stopwords)
    
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Lemmatize and filter
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) 
              for word in tokens 
              if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)


def preprocess_corpus(df: pd.DataFrame, 
                      text_column: str = 'Abstract',
                      title_column: str = 'Title') -> list:
    """
    Preprocess entire corpus combining title and abstract.
    
    Parameters:
        df: DataFrame with text columns
        text_column: Name of abstract column
        title_column: Name of title column
        
    Returns:
        List of preprocessed text strings
    """
    # Combine title and abstract
    combined_text = df[title_column].fillna('') + ' ' + df[text_column].fillna('')
    
    # Preprocess each document
    processed_docs = [preprocess_text(text) for text in combined_text]
    
    # Report statistics
    non_empty = sum(1 for doc in processed_docs if len(doc) > 0)
    print(f"Preprocessed {non_empty}/{len(processed_docs)} documents")
    
    return processed_docs


def perform_topic_modeling(documents: list,
                           n_topics: int = 5,
                           max_features: int = 1000,
                           random_state: int = 42) -> dict:
    """
    Perform LDA topic modeling on document corpus.
    
    Parameters:
        documents: List of preprocessed text strings
        n_topics: Number of topics to extract
        max_features: Maximum vocabulary size
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
        - lda_model: Fitted LDA model
        - vectorizer: Fitted CountVectorizer
        - doc_topic_matrix: Document-topic distributions
        - topics: Top words per topic
        - doc_term_matrix: Document-term matrix
    """
    # Filter empty documents
    valid_docs = [doc for doc in documents if len(doc.strip()) > 0]
    
    # Vectorization
    vectorizer = CountVectorizer(
        max_df=0.95,      # Ignore terms in >95% of docs
        min_df=2,         # Ignore terms in <2 docs
        max_features=max_features
    )
    doc_term_matrix = vectorizer.fit_transform(valid_docs)
    
    # LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        max_iter=20,
        learning_method='online',
        n_jobs=-1
    )
    doc_topic_matrix = lda_model.fit_transform(doc_term_matrix)
    
    # Extract top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    
    print(f"\n{'='*60}")
    print(f"LDA Topic Modeling Results ({n_topics} topics)")
    print(f"{'='*60}")
    
    for idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics[f'Topic_{idx+1}'] = top_words
        print(f"\nTopic {idx+1}: {', '.join(top_words)}")
    
    return {
        'lda_model': lda_model,
        'vectorizer': vectorizer,
        'doc_topic_matrix': doc_topic_matrix,
        'topics': topics,
        'doc_term_matrix': doc_term_matrix,
        'valid_docs': valid_docs
    }


def visualize_topics(topics: dict, save_path: str = 'topics_wordcloud.png'):
    """
    Visualize topic keywords as horizontal bar charts.
    
    Parameters:
        topics: Dictionary of topic names to word lists
        save_path: Path to save figure
    """
    n_topics = len(topics)
    fig, axes = plt.subplots(1, n_topics, figsize=(4*n_topics, 5))
    
    if n_topics == 1:
        axes = [axes]
    
    for idx, (topic_name, words) in enumerate(topics.items()):
        ax = axes[idx]
        y_pos = range(len(words))
        ax.barh(y_pos, range(len(words), 0, -1), color=f'C{idx}', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Relevance')
        ax.set_title(topic_name.replace('_', ' '))
    
    plt.suptitle('Topic Keywords', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def analyze_topic_evolution(df: pd.DataFrame,
                            doc_topic_matrix: np.ndarray,
                            date_column: str = 'Year') -> pd.DataFrame:
    """
    Analyze how topics evolve over time.
    
    Parameters:
        df: Original DataFrame with date column
        doc_topic_matrix: Document-topic distribution matrix
        date_column: Name of date column
        
    Returns:
        DataFrame with topic proportions by year
    """
    n_topics = doc_topic_matrix.shape[1]
    
    # Create dataframe with topic distributions
    topic_cols = [f'Topic_{i+1}' for i in range(n_topics)]
    df_topics = pd.DataFrame(doc_topic_matrix, columns=topic_cols)
    
    # Add year (handle both datetime and int)
    if hasattr(df[date_column].iloc[0], 'year'):
        df_topics['year'] = df[date_column].dt.year.values[:len(doc_topic_matrix)]
    else:
        df_topics['year'] = df[date_column].values[:len(doc_topic_matrix)]
    
    # Aggregate by year
    topic_evolution = df_topics.groupby('year')[topic_cols].mean()
    
    # Visualization: stacked area chart
    fig, ax = plt.subplots(figsize=(14, 8))
    topic_evolution.plot(kind='area', stacked=True, ax=ax, alpha=0.7,
                         colormap='tab10')
    
    ax.set_title('Topic Evolution Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Topic Proportion', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Topics')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('topic_evolution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return topic_evolution


def get_dominant_topic(doc_topic_matrix: np.ndarray) -> np.ndarray:
    """
    Get the dominant topic for each document.
    
    Parameters:
        doc_topic_matrix: Document-topic distribution matrix
        
    Returns:
        Array of dominant topic indices (1-indexed)
    """
    return np.argmax(doc_topic_matrix, axis=1) + 1


def find_representative_docs(df: pd.DataFrame,
                             doc_topic_matrix: np.ndarray,
                             topic_idx: int,
                             n_docs: int = 5) -> pd.DataFrame:
    """
    Find most representative documents for a given topic.
    
    Parameters:
        df: Original DataFrame
        doc_topic_matrix: Document-topic distribution matrix
        topic_idx: Topic index (1-indexed)
        n_docs: Number of documents to return
        
    Returns:
        DataFrame with top representative documents
    """
    # Get topic probabilities for this topic
    topic_probs = doc_topic_matrix[:, topic_idx - 1]
    
    # Get top document indices
    top_indices = topic_probs.argsort()[-n_docs:][::-1]
    
    # Extract documents
    result = df.iloc[top_indices][['Title', 'Year']].copy()
    result['Topic_Probability'] = topic_probs[top_indices]
    
    return result


# Example usage
if __name__ == "__main__":
    print("Topic Modeling Module - Chapter 12")
    print("Functions: preprocess_text(), perform_topic_modeling(), analyze_topic_evolution()")
    print("See README.md for complete documentation")
