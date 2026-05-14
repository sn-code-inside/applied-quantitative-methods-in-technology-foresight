# Chapter 8: Deep Learning Applications in Foresight

## Overview

> **Authors:**
> *Halime Özge Kabak, Assoc. Prof. Dr. Sila Ovgu Korkut** and Prof. Dr. Femin Yalcin**
>
> **Affiliations:** *RDT INGENIEROS, **Izmir Katip Celebi University, Faculty of Engineering and Architecture, Department of Engineering Sciences
> 
> **Email addresses:** eng.halimeozgekabak@gmail.com, silaovgu.korkut@ikcu.edu.tr, and femin.yalcin@ikcu.edu.tr 
>
> **ORCIDs:** https://orcid.org/0009-0009-7343-7610, https://orcid.org/0000-0003-4784-2013 and https://orcid.org/0000-0003-0602-9392

This folder contains the Python scripts, synthetic datasets, and reproducible analytical workflows developed for **Chapter 8: Deep Learning Applications in Foresight** in the book *Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches* (Springer).

The materials in this chapter demonstrate how deep learning and modern artificial intelligence methods can be applied to technology foresight, strategic intelligence, and emerging technology analysis. The repository includes practical examples covering:

- descriptive statistical analysis,
- uncertainty modeling and Monte Carlo simulation,
- hypothesis testing across technology domains,
- time-series forecasting and trend decomposition,
- lifecycle (S-curve) modeling,
- and retrospective technology assessment.

- Emerging technology detection
- Trend forecasting with deep learning
- Transformer-based text analytics
- Transfer learning applications
- Sentiment-driven foresight analytics
- Named Entity Recognition (NER)
- Strategic mapping using network analysis
- Representation learning with Sentence-BERT
- Time-series forecasting with LSTM and GRU architectures

The examples are designed for graduate students, researchers, analysts, and practitioners working in technology foresight, innovation analytics, computational social science, and strategic planning.

The repository is intended both for instructional use and for reproducible computational experimentation in technology foresight contexts.

---

## Repository Contents

*List and describe the files/scripts in this folder. Example:*
| File / Folder | Description |
|---------------|-------------|
| `ch8_project_1_energy_technology_trend_forecasting.py` | Panel GRU forecasting with rolling-origin backtesting on per-capita energy use data |
| `ch8_projects_2_3_4_NLP.py` | Text-based foresight analytics: clustering, LSTM trend forecasting, NER, and sentiment analysis using the CORD-19 dataset |
| `ch8_projects_5_6_transfer_learning.py` | Transfer learning case studies (feature extraction and fine-tuning) using MobileNetV2 on the Bean leaf disease dataset |
| `per-capita-energy-use.csv` | Energy usage data used by the energy technology trend forecasting project (Project 1) |
| `cord_19_dataset.zip` → `cord19_df.csv` | CORD-19 scientific literature snapshot used by the NLP projects (Projects 2, 3, and 4) |
| `bean_dataset.zip` → `angular_leaf_spot/`, `bean_rust/`, `healthy/` | Bean leaf disease image dataset used by the transfer learning projects (Projects 5 and 6) |
| `chapter_8_running_codes.ipynb` | Jupyter notebook to run all three scripts in the recommended order with environment verification |

## Generated Outputs

Each script automatically saves all outputs to dedicated folders. No manual path configuration is required.

| Output Directory | File | Description |
|-----------------|------|-------------|
| `outputs_ch8_project_1/` | `rolling_origin_metrics.csv` | Fold-level MAE and RMSE for baseline and Panel GRU across all rolling-origin test years |
| | `average_metrics.csv` | Average MAE and RMSE across all folds for both models |
| | `improvement_rates.csv` | Share of folds where the Panel GRU outperforms the persistence baseline |
| | `rmse_over_time.png` | Line plot of RMSE trends over rolling-origin test years |
| | `mae_over_time.png` | Line plot of MAE trends over rolling-origin test years |
| `outputs_ch8_project_2/` | `cluster_trends.png` | Topic cluster frequency by year visualizing emerging research themes |
| `outputs_ch8_project_3/` | `lstm_loss.png` | LSTM training and validation loss curve over epochs |
| `outputs_ch8_project_4/` | `entity_network.csv` | Entity co-occurrence network with source, target, and edge weight columns |
| `outputs_ch8_case_study/` | `sentiment_by_year.csv` | Yearly sentiment distribution of policy-related scientific abstracts |
| `outputs_ch8_projects_5_6/` | `run_metadata.json` | Experiment configuration, library versions, and determinism environment variables |
| `outputs_ch8_projects_5_6/outputs_ch8_project_5/feature_extraction/` | `best.keras` | Best model checkpoint saved during Project 5 feature extraction training |
| | `metrics.json` | Training history, test accuracy, and artifact paths for Project 5 |
| | `classification_report_project_5.txt` | Text classification report with precision, recall, F1-score, and support per class |
| | `classification_report_project_5.csv` | CSV version of the Project 5 classification report for table use |
| | `confusion_matrix_project_5.npy` | Raw confusion matrix values stored as a NumPy array |
| | `confusion_matrix_project_5.png` | Confusion matrix visualization for Project 5 test set evaluation |
| `outputs_ch8_projects_5_6/outputs_ch8_project_6/fine_tuning/` | `best.keras` | Best model checkpoint saved during Project 6 fine-tuning |
| | `metrics.json` | Training history, test accuracy, and artifact paths for Project 6 |
| | `classification_report_project_6.txt` | Text classification report with precision, recall, F1-score, and support per class |
| | `classification_report_project_6.csv` | CSV version of the Project 6 classification report for table use |
| | `confusion_matrix_project_6.npy` | Raw confusion matrix values stored as a NumPy array |
| | `confusion_matrix_project_6.png` | Confusion matrix visualization for Project 6 test set evaluation |

---

## Requirements

The scripts were developed and tested using:

```text
============================================================
ENVIRONMENT INFORMATION
============================================================
Python Version      : 3.10.20 
NumPy Version       : 2.2.5
Pandas Version      : 2.3.3
Scikit-learn Version: 1.7.1
Matplotlib Version  : 3.10.9
tqdm Version        : 4.67.3
PyTorch Version     : 2.12.0+cpu
CUDA Available      : False
TensorFlow Version  : 2.21.0
Keras Version       : 3.12.2
Transformers Version: 5.8.1
SentenceTransformers: 5.5.0
NetworkX Version    : 3.4.2
============================================================
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight.git
```

Navigate to the chapter folder:

```bash
cd applied-quantitative-methods-in-technology-foresight/chapter_8_deep_learning_applications_in_foresight

```

Install required libraries:

```bash
pip install sentence-transformers umap-learn transformers tf-keras ipykernel jupyter
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow
pip install tensorflow-datasets
```

---

## Usage

The scripts are designed to run independently, with the exception noted below.

---

### Environment Setup (Recommended)

Before running any script, create a clean Conda environment and install all required packages.

**1. Open Anaconda Prompt and create the environment:**

```bash
conda create -n foresight_env python=3.10 -y
conda activate foresight_env
```

**2. Install required packages:**

```bash
conda install pandas numpy scikit-learn matplotlib tqdm requests networkx -y

pip install tensorflow tf-keras
pip install "transformers>=4.41.0"
pip install sentence-transformers
pip install umap-learn
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> **GPU users:** Replace the last line with:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

**3. Register the environment as a Jupyter kernel:**

```bash
pip install ipykernel jupyter
python -m ipykernel install --user --name=foresight_env --display-name "Foresight Env"
```

**4. Launch Jupyter and select the kernel:**

```bash
jupyter notebook
```

In the notebook: *Kernel → Change Kernel → Foresight Env*

---

### Recommended Execution Order

1. `ch8_project_1_energy_technology_trend_forecasting.py`
2. `ch8_projects_2_3_4_NLP.py`
3. `ch8_projects_5_6_transfer_learning.py`

> **Dependency note:**
> - `ch8_project_1_energy_technology_trend_forecasting.py` — fully independent
> - `ch8_projects_2_3_4_NLP.py` — fully independent; the Case Study (Sentiment-Driven Policy Insight) is embedded in this script and depends on the CORD-19 data loaded at the beginning of the same script
> - `ch8_projects_5_6_transfer_learning.py` — fully independent
> - `chapter_8_running_codes.ipynb` — runs all three scripts in order; requires all data files to be present

**Example execution via terminal:**

```bash
python ch8_project_1_energy_technology_trend_forecasting.py
python ch8_projects_2_3_4_NLP.py
python ch8_projects_5_6_transfer_learning.py
```

**Example execution via Jupyter (recommended):**

```python
%run ch8_project_1_energy_technology_trend_forecasting.py
%run ch8_projects_2_3_4_NLP.py
%run ch8_projects_5_6_transfer_learnings.py
```

#### Each script automatically generates tables, figures, forecasts, statistical summaries, and visualization outputs inside dedicated output folders.
---

## Data Sources

The datasets used in this chapter are openly licensed and loaded either automatically at runtime or from local ZIP files.

| Dataset | Source | License | Loading Method |
|---------|--------|---------|----------------|
| CORD-19 Scientific Literature | [Semantic Scholar / Allen Institute for AI](https://allenai.org/data/cord-19) | CC BY 4.0 | Local ZIP (`cord_19_dataset.zip`) |
| Per-Capita Energy Use | [Our World in Data](https://ourworldindata.org/grapher/per-capita-energy-use) | CC BY 4.0 | Local ZIP (`per-capita-energy-use.csv`) |
| Bean Leaf Disease (ibean) | Makerere AI Lab | CC BY 4.0 | Local ZIP (`bean_dataset.zip`) |

### Local ZIP Files

The two ZIP-based datasets exceed GitHub's file size limit and are hosted externally. Download them from the link below and place them in the same folder as the scripts before running:

> [Download Datasets — Google Drive](https://drive.google.com/drive/folders/1080r21veYWBiHQS0SNC-0u--03hFTwHs?usp=sharing)

- **`cord_19_dataset.zip`** → contains `cord19_df.csv`
  Used by `ch8_projects_2_3_4_NLP.py` for topic modeling, trend detection, NER, and sentiment analysis.

- **`bean_dataset.zip`** → contains `angular_leaf_spot/`, `bean_rust/`, `healthy/`
  Used by `ch8_projects_5_6_transfer_learning.py` for transfer learning experiments. The script handles extraction and train/validation/test splitting automatically.

---

## Notes

The scripts emphasize pedagogical clarity and reproducibility over computational optimization.

-Emerging technology detection
-Trend forecasting with deep learning
-Transformer-based text analytics
-Transfer learning applications
-Sentiment-driven foresight analytics
-Named Entity Recognition (NER)
-Strategic mapping using network analysis
-Representation learning with Sentence-BERT
-Time-series forecasting with LSTM and GRU architectures

- The examples are intentionally simplified to support instructional use and self-study in deep learning applications for technology foresight.
- Transformer-based pipelines (NER, sentiment analysis, Sentence-BERT) may require significant memory and processing time on CPU-only machines. This is expected behavior and does not indicate an error.
- If `sentence-transformers` is unavailable or fails to load, `ch8_projects_2_3_4_NLP.py` automatically falls back to TF-IDF vectorization. Results may differ slightly from the Sentence-BERT baseline but remain analytically valid.
- If the `transformers` pipeline import fails, the NER (Project 4) and Sentiment Case Study modules will be skipped gracefully with a printed warning.
- LSTM and GRU training results may vary slightly across runs on the GPU due to nondeterministic CUDA kernels. For strict reproducibility, CPU execution is recommended.
- The rolling-origin backtesting in Project 1 is designed for pedagogical transparency; fold-level metrics may vary if the upstream Our World in Data source is updated over time.
- The Bean Leaf Disease transfer learning experiments (Projects 5 and 6) use a fixed random seed for deterministic train/validation/test splitting. Results are reproducible within the same environment and package versions.
- The chapter combines quantitative rigor with critical reflection on the limitations of deep learning forecasting in technology foresight contexts, including data scarcity, domain shift, and model interpretability.
- Minor numerical differences may occur across different operating systems, hardware configurations, and package versions. These differences are common in deep learning workflows and do not invalidate the reported results.

## Reproducibility

The implementations emphasize reproducibility through:

-Fixed random seeds across Python, NumPy, PyTorch, and TensorFlow (seed=42)
-Deterministic environment flags set before runtime: TF_DETERMINISTIC_OPS, TF_CUDNN_DETERMINISTIC, PYTHONHASHSEED
-Stable sorting before sequence construction
-Time-based validation splits with no random data leakage
-Rolling-origin backtesting for temporal evaluation
-Local dataset loading as a reproducible alternative to URL-based loading
-Saved model checkpoints, evaluation metrics, training histories, and configuration files

Minor numerical differences may occur across different operating systems, hardware configurations, and package versions. Some GPU operations may remain nondeterministic depending on CUDA and cuDNN versions. For strict reproducibility, CPU execution is recommended.

---

*Part of: **Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches** — Springer*  
*Repository: https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight/*

