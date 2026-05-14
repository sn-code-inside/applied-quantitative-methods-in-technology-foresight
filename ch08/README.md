# Chapter 8: Deep Learning Applications in Foresight

## Overview

> **Authors:** Halime Özge Kabak · Sıla Övgü Korkut · Femin Yalçın

This repository contains the source code, datasets, and supplementary materials developed for Chapter 8, *Deep Learning Applications in Foresight*, from the book *Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches*.

The implementations demonstrate how deep learning and artificial intelligence techniques can be applied to technology foresight, strategic intelligence, and innovation analytics. The chapter focuses on practical and reproducible workflows involving transformer-based NLP methods, transfer learning, time-series forecasting, clustering, sentiment analysis, and strategic mapping.

The repository is designed for graduate students, researchers, foresight analysts, and practitioners interested in applying AI-driven approaches to technology forecasting and emerging technology detection.

---

## Repository Contents

| File | Description |
|------|-------------|
| `NLP_Project_Assignments.py` | Main NLP-based foresight analytics implementation including clustering, LSTM forecasting, sentiment analysis, and entity-based strategic mapping. |
| `Energy_Technology_Trend_Forecasting.py` | GRU-based panel forecasting framework for energy technology trend analysis using rolling-origin temporal validation. |
| `transfer_learning_case_studies.py` | Transfer learning case studies using TensorFlow/Keras and MobileNetV2 for feature extraction and fine-tuning workflows. |
| `cord_19_dataset.zip` | Local archival snapshot of the CORD-19 dataset containing `cord19_df.csv`. |
| `entity_network.csv` | Output file generated from the entity-based strategic mapping workflow. |
| `sentiment_by_year.csv` | Output file generated from sentiment-driven foresight analytics. |

---

## Requirements

The implementations were tested using the following environment:

```text
Python 3.13.5
NumPy 2.1.3
Pandas 2.3.3
Scikit-learn 1.7.2
Matplotlib 3.10.7
tqdm 4.67.1
PyTorch 2.11.0+cpu
TensorFlow 2.20.0
Keras 3.11.3
Transformers 4.57.6
NetworkX 3.5
```
---

However, for the most stable execution of transformer-based pipelines, the recommended Python version is:
 
```text Python 3.10 or Python 3.11 ```

Optional dependency:

sentence-transformers

If sentence-transformers is not installed, the NLP implementation automatically falls back to TF-IDF vectorization.

## Installation

It is recommended to create a clean environment before running the scripts.

### Recommended Conda Environment

```bash
conda create -n chapter8 python=3.11 -y
conda activate chapter8
```

### Install Required Packages

```bash
Siskom burada Özge'nin partı için gerekenler:
pip install tensorflow
pip install tensorflow-datasets


pip install -U pandas numpy scikit-learn matplotlib tqdm requests networkx
pip install -U tensorflow keras tf-keras
pip install -U torch
pip install "transformers>=4.41.0,<5.0.0"
```

### Optional Installation

```bash
pip install sentence-transformers
```

If `sentence-transformers` is unavailable or fails to load, the code will automatically use TF-IDF vectorization as a fallback.

---

# Usage

The scripts can be run independently depending on the application.

## 1. Run NLP-Based Foresight Analytics

```bash
python NLP_Project_Assignments.py
```

This script performs:

- emerging technology detection,
- embedding-based clustering,
- LSTM trend forecasting,
- entity-based strategic mapping,
- sentiment-driven policy insight analysis.

---

## 2. Run Energy Technology Forecasting

```bash
python Energy_Technology_Trend_Forecasting.py
```

This script performs:

- rolling-origin forecasting,
- panel GRU modeling,
- country-wise normalization,
- persistence baseline comparison,
- temporal performance evaluation.

---

## 3. Run Transfer Learning Case Studies

```bash
python transfer_learning_case_studies.py
```

This script performs:

- Case Study A: feature extraction using a frozen MobileNetV2 backbone,
- Case Study B: fine-tuning selected layers of the pretrained backbone.

### Recommended Execution Order

```text
1. NLP_Project_Assignments.py
2. Energy_Technology_Trend_Forecasting.py
3. transfer_learning_case_studies.py
```

---

# Data Sources

## CORD-19 Dataset

The NLP-based foresight workflow uses the CORD-19 scientific literature dataset.

The primary implementation downloads the dataset directly from the official online source using:

```python
load_cord19_from_url()
```

For offline or archival reproducibility, a local ZIP-based snapshot is also provided:

```python
load_cord19_from_zip()
```

### Local Archive

```text
cord_19_dataset.zip
```

### Included File

```text
cord19_df.csv
```

CORD-19 is an open scientific literature dataset originally released by the Allen Institute for AI and Semantic Scholar.

### License

```text
CC BY 4.0
```

---

## Our World in Data

The energy forecasting implementation uses an online energy-use-per-capita dataset from Our World in Data.

The dataset is retrieved directly in the script using a URL-based loading strategy.

---

# Notes

- The scripts are designed to support the computational examples discussed in Chapter 8.
- The NLP script includes fallback mechanisms to improve robustness across different computing environments.
- If `sentence-transformers` is unavailable, TF-IDF vectorization is used instead.
- If transformer-based `pipeline` import fails, the related NER or sentiment module may be skipped depending on the local environment.
- For the most stable execution of `transformers.pipeline`, Python 3.10 or Python 3.11 is recommended.
- The tested environment used CPU-only PyTorch execution.
- CUDA/GPU acceleration is supported if compatible CUDA-enabled PyTorch and TensorFlow installations are available.
- Online datasets may change over time. For exact reproducibility, local snapshots such as `cord_19_dataset.zip` should be used.
- Minor numerical differences may occur across operating systems, hardware, package versions, and random initialization settings.
- Some deep learning operations may remain nondeterministic on GPU depending on CUDA/cuDNN configuration.

---

# Generated Outputs

| Output File | Description |
|---|---|
| `cluster_trends.png` | Visualization of cluster frequency trends by year. |
| `lstm_loss.png` | LSTM training and validation loss plot. |
| `entity_network.csv` | Entity co-occurrence network generated from NER outputs. |
| `sentiment_by_year.csv` | Yearly sentiment distribution generated from policy-related abstracts. |
| `outputs_beans_transfer_learning/` | Output directory generated by transfer learning case studies. |
| `run_metadata.json` | Metadata file containing configuration and library version information for reproducibility. |

---

# Reproducibility

The code emphasizes reproducibility through:

- fixed random seeds,
- deterministic preprocessing,
- stable sorting procedures,
- temporal validation,
- rolling-origin evaluation,
- local and URL-based data loading alternatives,
- saved configuration files,
- saved model outputs and evaluation artifacts,
- version reporting for core libraries.

For archival reproducibility, users are encouraged to:

1. use the local dataset snapshots,
2. record Python and package versions,
3. avoid changing random seeds,
4. use CPU execution when strict reproducibility is required,
5. save generated outputs with timestamps.
