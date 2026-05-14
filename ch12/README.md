# Chapter 12: Data-Led Technology Roadmapping

## Overview

**Authors:** Arif Soyler, Prof. Dr. Serhat Burmaoglu, and Asst. Prof. Dr. Esra Dundar Aravacik

**Affiliations:**
- Izmir Bakircay University, Institute of Graduate Studies, Department of Health Management (PhD Candidate) — *Arif Soyler*
- Izmir Katip Celebi University, Faculty of Economics and Administrative Sciences, Department of Department of Data Science and Analytics *Serhat Burmaoglu*
- Izmir Katip Celebi University, Faculty of Economics and Administrative Sciences, Department of Health Management — *Esra Dundar Aravacik*

**Email addresses:** arifsoyler@gmail.com, serhatburmaoglu@gmail.com, and esra.dundar@hotmail.com

**ORCIDs:** [https://orcid.org/0000-0001-7699-6316](https://orcid.org/0000-0001-7699-6316), [https://orcid.org/0000-0002-5537-6887](https://orcid.org/0000-0002-5537-6887), and [https://orcid.org/0000-0002-6504-6283](https://orcid.org/0000-0002-6504-6283)

---

This folder contains the Python modules, synthetic datasets, and reproducible analytical workflows developed for **Chapter 12: Data-Led Technology Roadmapping** in the book *Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches* (Springer).

The chapter presents a comprehensive framework for data-led technology roadmapping, integrating bibliometric analysis, patent analytics, text mining, and network analysis methods. Complete Python implementations are provided for trend analysis, topic modeling using Latent Dirichlet Allocation, co-occurrence network construction, and technology maturity assessment. The healthcare smart contracts case study, analyzing 4,156 publications and 10,204 patents, demonstrates a sophisticated eight-stage analytical pipeline combining traditional and transformer-based methods.

The scripts included in this repository support the examples and case study presented throughout the chapter. They demonstrate reproducible implementations of:

- data loading and preprocessing for publication and patent databases,
- time series analysis and ARIMA-based technology trend forecasting,
- topic modeling using Latent Dirichlet Allocation (LDA),
- keyword co-occurrence network analysis with centrality measures and community detection,
- and technology roadmap generation and visualization.

The repository is intended both for instructional use and for reproducible computational experimentation in technology foresight and roadmapping contexts.

## Repository Contents

| File / Folder | Description |
|---|---|
| `code/data_loading.py` | Data loading and preprocessing module for publication and patent datasets |
| `code/time_series.py` | Time series analysis and forecasting module including ARIMA modeling and S-curve fitting |
| `code/topic_modeling.py` | Topic modeling and text analysis module using Latent Dirichlet Allocation (LDA) |
| `code/network_analysis.py` | Network analysis module for keyword co-occurrence, centrality measures, and community detection |
| `code/roadmap_generation.py` | Roadmap generation and visualization module |
| `data/synthetic_quantum_combined.csv` | Synthetic combined dataset (publications and patents) for quantum computing technology |
| `data/synthetic_battery_patents.csv` | Synthetic patent dataset for battery technology used in Example 1 |
| `data/synthetic_ai_healthcare_pubs.csv` | Synthetic publications dataset for AI in healthcare used in Example 2 |
| `DATA_REPLICATION_GUIDE.md` | Step-by-step guide for replicating analyses with real-world data from Lens.org and PubMed |
| `requirements.txt` | Python package dependencies for the project |

## Requirements

The scripts were developed and tested using:

- Python 3.8+
- numpy
- pandas
- matplotlib
- scipy
- statsmodels
- scikit-learn
- gensim
- networkx
- nltk

## Installation

Clone the repository:

```bash
git clone https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight.git
```

Navigate to the chapter folder:

```bash
cd applied-quantitative-methods-in-technology-foresight/ch12
```

Install required libraries:

```bash
pip install -r requirements.txt
```

## Usage

The code is organized as modular Python scripts designed to be executed sequentially as an analytical pipeline. Recommended execution order:

1. `code/data_loading.py` — Load and preprocess raw publication/patent data
2. `code/time_series.py` — Perform temporal trend analysis and forecasting
3. `code/topic_modeling.py` — Extract latent topics from document corpora
4. `code/network_analysis.py` — Construct and analyze co-occurrence networks
5. `code/roadmap_generation.py` — Generate integrated technology roadmap visualizations

Example execution:

```bash
python code/data_loading.py
python code/time_series.py
python code/topic_modeling.py
python code/network_analysis.py
python code/roadmap_generation.py
```

## Data Sources

The datasets included in this repository are synthetic educational datasets designed to reproduce the analytical workflows presented in the chapter:

- **Battery patents dataset** (Example 1): Synthetic patent records for battery technology trend analysis
- **AI healthcare publications dataset** (Example 2): Synthetic publication records for topic evolution analysis
- **Quantum computing combined dataset**: Synthetic combined publication and patent data for integrated roadmapping

The case study references data from [Lens.org](https://www.lens.org/), which provides free access to patent and scholarly publications. The search query used was: `((smart contract* OR blockchain*) AND (health* OR medical* OR clinical*))`. Data extraction was performed on August 15, 2024.

To ensure reproducibility and avoid licensing restrictions, all distributed datasets are simplified synthetic representations calibrated to reflect plausible technological dynamics rather than authoritative real-world counts. See `DATA_REPLICATION_GUIDE.md` for detailed instructions on replicating analyses with real-world data.

## Notes

- The scripts emphasize pedagogical clarity and reproducibility over computational optimization.
- The examples are intentionally simplified to support instructional use and self-study at the graduate level.
- The modular code architecture allows readers to adapt individual components (e.g., topic modeling, network analysis) for their own technology foresight projects.
- The chapter combines quantitative rigor with critical reflection on the limitations of data-driven roadmapping, emphasizing the importance of complementary qualitative and participatory foresight methods.
- The case study reveals a reversed innovation pattern where patents precede publications by 8.12 years on average, challenging conventional linear models of knowledge transfer.

**Part of:** *Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches* — Springer

**Repository:** https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight/


For questions about Chapter 12:

Arif Soyler
Izmir Bakırcay Celebi University
Department of Health Management
Email: arifsoyler@gmail.com
