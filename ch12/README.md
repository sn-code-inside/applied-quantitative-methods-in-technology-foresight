# Data-Led Technology Roadmapping

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Code implementations and datasets for **Chapter 12: Data-Led Technology Roadmapping** from the book *Applied Quantitative Methods in Technology Foresight* (Springer).

## 📖 Overview

This repository provides complete, production-ready Python implementations for data-led technology roadmapping methodologies including:

- **Bibliometric analysis** of scientific publications
- **Patent analytics** for technology assessment
- **Topic modeling** using Latent Dirichlet Allocation (LDA)
- **Network analysis** for technology relationship mapping
- **Time series forecasting** with S-curve fitting and ARIMA
- **Technology maturity assessment** frameworks

## 📁 Repository Structure
```
data-led-technology-roadmapping/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Python dependencies
├── code/
│   ├── data_loading.py       # Data import and preprocessing
│   ├── time_series.py        # Trend analysis and forecasting
│   ├── topic_modeling.py     # LDA topic modeling
│   ├── network_analysis.py   # Co-occurrence networks
│   └── roadmap_generation.py # Integrated roadmap creation
└── data/
    ├── synthetic_battery_patents.csv      # Example 1: Battery technology
    ├── synthetic_ai_healthcare_pubs.csv   # Example 2: AI in healthcare
    └── synthetic_quantum_combined.csv     # Example 3: Quantum computing
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/arfsylr/data-led-technology-roadmapping.git
cd data-led-technology-roadmapping
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 5. Run Example Analysis
```python
from code.data_loading import load_patent_data
from code.time_series import analyze_temporal_trends
from code.topic_modeling import preprocess_corpus, perform_topic_modeling
from code.network_analysis import create_cooccurrence_network
from code.roadmap_generation import assess_technology_maturity

# Load data
df = load_patent_data('data/synthetic_battery_patents.csv')

# Analyze trends
trends, growth = analyze_temporal_trends(df, date_column='Filing_Date')

# Topic modeling
docs = preprocess_corpus(df, text_column='Abstract', title_column='Title')
topics = perform_topic_modeling(docs, n_topics=5)

# Network analysis
G = create_cooccurrence_network(df, keyword_column='Keywords', separator=';')

# Maturity assessment
maturity = assess_technology_maturity(df, date_column='Filing_Date')
```

## 📊 Datasets

All datasets are **synthetic** (CC0 license) designed to demonstrate analytical methods while preserving realistic patterns.

| Dataset | Records | Description | Chapter Section |
|---------|---------|-------------|-----------------|
| `synthetic_battery_patents.csv` | 35 | Battery technology patents (2015-2024) | Example 1 (Section 3.1) |
| `synthetic_ai_healthcare_pubs.csv` | 40 | AI healthcare publications (2016-2024) | Example 2 (Section 3.2) |
| `synthetic_quantum_combined.csv` | 42 | Quantum computing (patents + publications) | Example 3 (Section 3.3) |

### Using Your Own Data

To use real data from bibliometric databases:

1. **Scopus/Web of Science**: Export as CSV with Title, Abstract, Year, Authors, Keywords
2. **Lens.org**: Export with Title, Abstract, Filing_Date, IPC_Class, Assignee
3. **Ensure proper licensing** before redistribution

## 📚 Module Documentation

### data_loading.py
- `load_publication_data()`: Load and preprocess publication CSV/Excel files
- `load_patent_data()`: Load and preprocess patent data with IPC parsing
- `combine_publication_patent_data()`: Merge datasets for integrated analysis

### time_series.py
- `analyze_temporal_trends()`: Calculate growth rates and visualize trends
- `fit_scurve()`: Fit logistic S-curve for maturity assessment
- `forecast_technology_trend()`: ARIMA-based forecasting with confidence intervals

### topic_modeling.py
- `preprocess_text()`: Text cleaning, stopword removal, lemmatization
- `perform_topic_modeling()`: LDA topic extraction with visualization
- `analyze_topic_evolution()`: Track topic proportions over time

### network_analysis.py
- `create_cooccurrence_network()`: Build keyword co-occurrence graph
- `analyze_network_centrality()`: Calculate degree, betweenness, eigenvector centrality
- `detect_communities()`: Identify technology clusters using modularity optimization

### roadmap_generation.py
- `assess_technology_maturity()`: Classify as Emerging/Growth/Mature/Declining
- `create_technology_roadmap()`: Generate integrated multi-panel visualization
- `generate_roadmap_report()`: Create text-based strategic report

## 📖 Citation

If you use this code in your research, please cite:
```bibtex
@incollection{soyler2026roadmapping,
  title={Data-Led Technology Roadmapping},
  author={Söyler, Arif and Aravacik, Esra Dundar},
  booktitle={Applied Quantitative Methods in Technology Foresight},
  editor={Burmaoglu, Serhat},
  publisher={Springer},
  year={2026},
  chapter={12}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Arif Söyler** - Izmir Bakircay University, Department of Health Management (PhD Candidate)
- **Asst. Prof. Dr. Esra Dundar Aravacik** - Izmir Katip Celebi University, Faculty of Economics and Administrative Sciences

## 🙏 Acknowledgments

- Prof. Dr. Serhat Burmaoglu (Editor)
- Springer Publishing
- All contributors to the open-source libraries used in this project

---

**Questions or Issues?** Please open an issue on this repository.
