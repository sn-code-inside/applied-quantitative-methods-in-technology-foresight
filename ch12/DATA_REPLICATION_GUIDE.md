# Data Replication Guide for Chapter 12: Data-Led Technology Roadmapping

## Overview

This guide provides complete instructions for replicating the healthcare smart contracts case study presented in Chapter 12. Due to Lens.org terms of service, raw data cannot be redistributed. However, this guide enables readers to reproduce identical datasets through their own Lens.org accounts.

**Reference Publication:** Soyler, A., Burmaoglu, S., & Dundar Aravacik, E. (2025). [Divergent priorities, convergent goals semantic analysis of publications and patents in healthcare smart contracts]. *Scientometrics*. DOI: 10.1007/s11192-025-05457-1

---

## Part 1: Data Access and Download

### Step 1: Create Lens.org Account

1. Navigate to https://www.lens.org
2. Click "Sign Up" (top right corner)
3. Complete free registration with email verification
4. Log in to your account

### Step 2: Download Scholarly Publications

1. From the main page, select **"Scholarly Works"** in the left menu
2. In the search box, enter the following query exactly:

```
((smart contract* OR blockchain*) AND (health* OR medical* OR clinical*))
```

3. Apply the following filters:
   - **Publication Date:** Custom range → 2001-01-01 to 2024-12-31
   - **Document Type:** All (or select: Journal Article, Conference Paper, Preprint)

4. Click **"Search"**

5. **Export Settings:**
   - Click "Select All" to select all results
   - Click **"Export"** button
   - Select Format: **CSV**
   - Select the following fields:
     - Lens ID
     - Title
     - Abstract
     - Publication Year
     - Authors
     - Source Title (Journal Name)
     - Keywords
     - DOI
     - Citation Count
     - Publication Type
   - Click "Export"
   - Save file as: `healthcare_smart_contracts_publications.csv`

**Expected Result:** Approximately 4,156 records (±5% variance due to database updates)

### Step 3: Download Patent Data

1. From the main page, select **"Patents"** in the left menu
2. In the search box, enter the same query:

```
((smart contract* OR blockchain*) AND (health* OR medical* OR clinical*))
```

3. Apply the following filters:
   - **Filing Date:** Custom range → 2001-01-01 to 2024-12-31
   - **Jurisdiction:** All (for comprehensive coverage)

4. Click **"Search"**

5. **Export Settings:**
   - Click "Select All" to select all results
   - Click **"Export"** button
   - Select Format: **CSV**
   - Select the following fields:
     - Lens ID
     - Title
     - Abstract
     - Filing Date
     - Publication Date
     - Applicants/Assignees
     - Inventors
     - IPC Classifications
     - CPC Classifications
     - Priority Country
     - Legal Status
   - Click "Export"
   - Save file as: `healthcare_smart_contracts_patents.csv`

**Expected Result:** Approximately 10,204 records (±5% variance due to database updates)

---

## Part 2: File Organization (Path Structure)

After downloading, organize your files according to the following directory structure:

```
chapter12/
├── README.md
├── requirements.txt
├── DATA_REPLICATION_GUIDE.md          ← This file
│
├── code/
│   ├── data_loading.py                 # Section 4.2 functions
│   ├── time_series.py                  # Section 4.3 functions
│   ├── topic_modeling.py               # Section 4.4 functions
│   ├── network_analysis.py             # Section 4.5 functions
│   ├── roadmap_generation.py           # Section 4.6 functions
│   └── complete_workflow.py            # Integrated 8-stage pipeline
│
├── data/
│   ├── synthetic/                      # CC0 licensed synthetic datasets
│   │   ├── synthetic_battery_patents.csv
│   │   ├── synthetic_ai_healthcare_pubs.csv
│   │   └── synthetic_quantum_combined.csv
│   │
│   └── user_extracted/                 # YOUR downloaded data goes here
│       ├── healthcare_smart_contracts_publications.csv
│       └── healthcare_smart_contracts_patents.csv
│
├── outputs/
│   ├── figures/                        # Generated visualizations
│   │   ├── temporal_trends.png
│   │   ├── topic_evolution.png
│   │   ├── cooccurrence_network.png
│   │   └── integrated_roadmap.png
│   │
│   └── results/                        # Analysis outputs
│       ├── topic_distributions.csv
│       ├── centrality_measures.csv
│       └── maturity_assessment.json
│
└── documentation/
    ├── data_dictionary.md              # Field descriptions
    ├── methodology_notes.md            # Extended guidance
    └── troubleshooting.md              # Common issues
```

### Important Path Notes:

- Place your downloaded Lens.org data in `data/user_extracted/`
- The code expects files at these exact paths:
  - `data/user_extracted/healthcare_smart_contracts_publications.csv`
  - `data/user_extracted/healthcare_smart_contracts_patents.csv`
- Output files will be automatically saved to `outputs/` directory

---

## Part 3: Eight-Stage Analytical Pipeline

The case study employs an eight-stage computational pipeline as described in Section 5.2.3 of the chapter. Below is the complete methodology with code references:

### Stage 1: Text Preprocessing
**Purpose:** Prepare raw text for analysis  
**Code:** `code/topic_modeling.py` → `preprocess_text()`

```python
# Key operations:
# - Concatenate title + abstract
# - Lowercase conversion
# - Remove punctuation and special characters
# - Tokenization
# - Stopword removal (English + domain-specific)
# - Lemmatization using WordNet
```

### Stage 2: TF-IDF Vectorization
**Purpose:** Convert text to numerical features  
**Code:** `code/topic_modeling.py` → `create_tfidf_matrix()`

```python
# Parameters used:
# - max_df = 0.85 (remove terms in >85% of documents)
# - min_df = 5 (require term in ≥5 documents)
# - max_features = 5000
# - Separate matrices for publications and patents
```

### Stage 3: Latent Dirichlet Allocation (LDA)
**Purpose:** Identify latent topic structure  
**Code:** `code/topic_modeling.py` → `perform_topic_modeling()`

```python
# Parameters used:
# - n_components = 10 (optimal k via coherence score)
# - random_state = 42 (reproducibility)
# - max_iter = 20
# - Separate models for publications and patents
```

### Stage 4: SciBERT Semantic Embeddings
**Purpose:** Capture contextual semantic relationships  
**Code:** `code/topic_modeling.py` → `generate_scibert_embeddings()`

```python
# Configuration:
# - Model: allenai/scibert_scivocab_uncased
# - Output: 768-dimensional dense vectors
# - Pooling: Mean of last layer token embeddings
# - Dimensionality reduction: PCA to 128 dimensions
```

### Stage 5: K-Means Clustering
**Purpose:** Identify high-level domain structures  
**Code:** `code/topic_modeling.py` → `perform_clustering()`

```python
# Parameters used:
# - n_clusters = 2 (determined via elbow method)
# - Clustering on SciBERT embeddings
# - Results: Cluster 1 (20.6% technical), Cluster 2 (79.4% application)
```

### Stage 6: Temporal Trend Analysis
**Purpose:** Analyze innovation dynamics over time  
**Code:** `code/time_series.py` → `analyze_temporal_trends()`

```python
# Metrics calculated:
# - Annual document counts
# - CAGR (Compound Annual Growth Rate)
# - Growth rate comparisons
# - Inflection point identification
```

### Stage 7: Cross-Corpus Semantic Similarity
**Purpose:** Measure publication-patent alignment  
**Code:** `code/roadmap_generation.py` → `calculate_cross_corpus_similarity()`

```python
# Method:
# - Cosine similarity between SciBERT embeddings
# - Temporal lag calculation per document pair
# - Mean document similarity lag: -1.70 years
```

### Stage 8: Keyword Emergence Lag Assessment
**Purpose:** Track terminology transfer between corpora  
**Code:** `code/roadmap_generation.py` → `analyze_keyword_emergence()`

```python
# Method:
# - Extract author/applicant keywords
# - Track first appearance year (threshold: ≥10 occurrences)
# - Calculate cross-corpus lag
# - Mean keyword emergence lag: -8.12 years
```

---

## Part 4: Running the Complete Analysis

### Prerequisites

```bash
# Clone repository
git clone https://github.com/arfsylr/data-led-technology-roadmapping.git
cd data-led-technology-roadmapping/chapter12

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Execute Analysis

```bash
# Option 1: Run complete pipeline with your extracted data
python code/complete_workflow.py \
    --publications data/user_extracted/healthcare_smart_contracts_publications.csv \
    --patents data/user_extracted/healthcare_smart_contracts_patents.csv \
    --output outputs/ \
    --topics 10

# Option 2: Run with synthetic data (for testing)
python code/complete_workflow.py \
    --publications data/synthetic/synthetic_ai_healthcare_pubs.csv \
    --output outputs/ \
    --topics 5

# Option 3: Interactive Jupyter notebook
jupyter notebook notebooks/chapter12_case_study.ipynb
```

---

## Part 5: Expected Results and Validation

### Sample Characteristics (for validation)

| Metric | Publications | Patents |
|--------|-------------|---------|
| Total Records | ~4,156 | ~10,204 |
| Date Range | 2001-2025 | 2001-2025 |
| Peak Year | 2022-2023 | 2021-2022 |
| CAGR (2016-2024) | 47.3% | 38.6% |

### Key Findings to Replicate

1. **Reversed Knowledge Transfer:** Patents precede publications by ~8.12 years in keyword emergence
2. **Complementary Specialization:** Publications (79.4% application-focused) vs Patents (67.3% infrastructure-focused)
3. **Publication-to-Patent Ratio:** Approximately 1:2.45

### Validation Checklist

- [ ] Publication count within ±5% of 4,156
- [ ] Patent count within ±5% of 10,204
- [ ] Topic coherence scores > 0.4
- [ ] Keyword emergence lag negative (patents first)
- [ ] Two-cluster structure confirmed via silhouette analysis

---

## Part 6: Troubleshooting

### Common Issues

**Issue:** Different record counts than expected  
**Solution:** Lens.org continuously updates. Small variations (±5%) are normal. Ensure date filters are correctly applied.

**Issue:** Export limit reached  
**Solution:** Lens.org may limit exports. Try exporting in batches by year or split the query.

**Issue:** Missing fields in export  
**Solution:** Some records may have incomplete metadata. The preprocessing code handles missing values by filling abstracts with titles.

**Issue:** Memory errors with SciBERT  
**Solution:** Process documents in batches of 1000. Reduce max_features in TF-IDF vectorization.

### Support

For technical questions about this replication guide, please:
1. Check the Issues section of the GitHub repository
2. Refer to the chapter's Further Reading section
3. Contact the authors via the information provided in the chapter

---

## License and Citation

### Data License
- Lens.org data: Subject to Lens.org Terms of Service (free access, no bulk redistribution)
- Synthetic datasets: CC0 Public Domain

### Citation
If you use this methodology or code in your research, please cite:

```bibtex
@incollection{burmaoglu2026dataled,
  author    = {Burmaoglu, Serhat and Soyler, Arif and Dundar Aravacik, Esra},
  title     = {Data-Led Technology Roadmapping},
  booktitle = {Applied Quantitative Methods in Technology Foresight},
  publisher = {Springer},
  year      = {2026},
  chapter   = {12}
}
```

---

*Last updated: January 2026*  
*Guide version: 1.0*
