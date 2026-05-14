# Chapter 3: Statistical Foundations for Technology Analysis

## Overview

> **Authors:**
> Prof. Dr. Femin Yalcin,
> Assoc. Prof. Dr. Sila Ovgu Korkut 

This folder contains the Python scripts, synthetic datasets, and reproducible analytical workflows developed for **Chapter 3: Statistical Foundations for Technology Analysis** in the book *Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches* (Springer).

The chapter introduces the statistical foundations required for evidence-based technology foresight and innovation analysis. Through descriptive statistics, probabilistic modeling, inferential statistics, simulation, and time-series forecasting, readers learn how to transform raw technological indicators into interpretable and actionable foresight insights.

The scripts included in this repository support the examples and case study presented throughout the chapter. They demonstrate reproducible implementations of:

- descriptive statistical analysis,
- uncertainty modeling and Monte Carlo simulation,
- hypothesis testing across technology domains,
- time-series forecasting and trend decomposition,
- lifecycle (S-curve) modeling,
- and retrospective technology assessment.

The repository is intended both for instructional use and for reproducible computational experimentation in technology foresight contexts.

---

## Repository Contents

*List and describe the files/scripts in this folder. Example:*

| File / Folder | Description |
|------|-------------|
| `ch3_example_1_descriptives.py` | Descriptive statistical analysis of patent indicators including normalization, outlier detection, skewness, kurtosis, and visualization |
| `ch3_example_2_simulation.py` | Monte Carlo simulation of technology adoption using a Bass-type diffusion model with uncertainty bands |
| `ch3_example_3_hypothesis_testing.py` | Hypothesis testing and effect-size analysis comparing annual patent growth rates across technology domains |
| `ch3_example_4_time_series.py` | Time-series decomposition and forecasting of publication counts using ARIMA and ETS models |
| `ch3_CRISPR_case_study.py` | Integrated retrospective technology foresight case study on CRISPR gene-editing technology |
| `patent_trends.csv` | Synthetic patent-count dataset used in Examples 3.1 and 3.3 |
| `innovation_adoption_sim.csv` | Synthetic technology-adoption dataset used in Example 3.2 |
| `pub_trends.csv` | Synthetic publication-count dataset used in Example 3.4 |
| `outputs_ch3_example_1/` | Automatically generated outputs for Example 3.1 |
| `outputs_ch3_example_2/` | Automatically generated outputs for Example 3.2 |
| `outputs_ch3_example_3/` | Automatically generated outputs for Example 3.3 |
| `outputs_ch3_example_4/` | Automatically generated outputs for Example 3.4 |
| `outputs_ch3_CRISPR_case_study/` | Automatically generated outputs for the CRISPR case study |

---

## Requirements

The scripts were developed and tested using:

```text
Python 3.11+
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
scipy >= 1.10
statsmodels >= 0.14
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight.git
```

Navigate to the chapter folder:

```bash
cd applied-quantitative-methods-in-technology-foresight/chapter_3_statistical_foundations
```

Install required libraries:

```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install scipy
pip install statsmodels
```

---

## Usage

The scripts are designed to run independently. Recommended execution order:

1. `ch3_example_1_descriptives.py`
2. `ch3_example_2_simulation.py`
3. `ch3_example_3_hypothesis_testing.py`
4. `ch3_example_4_time_series.py`
5. `ch3_CRISPR_case_study.py`

Example execution:

```bash
python ch3_example_1_descriptives.py
```

Each script automatically generates:
- tables,
- figures,
- forecasts,
- statistical summaries,
- and visualization outputs

inside dedicated output folders.

---

## Data Sources

The datasets included in this repository are primarily **synthetic educational datasets** designed to reproduce the analytical workflows presented in the chapter.

The CRISPR case study references publicly accessible scientific and innovation databases conceptually consistent with:

- [PubMed](https://pubmed.ncbi.nlm.nih.gov)
- [Web of Science](https://www.webofscience.com)
- [USPTO Patent Search](https://www.uspto.gov/patents/search)
- [WIPO IP Statistics Data Center](https://www3.wipo.int/ipstats)
- [ClinicalTrials.gov](https://clinicaltrials.gov)
- [The Lens](https://www.lens.org)
- [OpenAlex](https://openalex.org)

To ensure reproducibility and avoid licensing restrictions, all distributed datasets are simplified synthetic representations calibrated to reflect plausible technological dynamics rather than authoritative real-world counts.

---

## Notes

- The scripts emphasize pedagogical clarity and reproducibility over computational optimization.
- The examples are intentionally simplified to support instructional use and self-study.
- Technology indicators frequently violate strict statistical assumptions; therefore, the chapter emphasizes diagnostic checking, robust interpretation, and methodological transparency.
- Annual technology time series may generate warnings related to non-stationary starting parameters or unsupported indices in some ARIMA implementations. These warnings are common in small annual datasets and typically do not invalidate the fitted models.
- The chapter combines quantitative rigor with critical reflection on the limitations of statistical forecasting in technology foresight contexts.
- The CRISPR case study demonstrates how statistical methods can support retrospective technology assessment while highlighting the importance of complementary qualitative and participatory foresight methods.

---

*Part of: **Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches** — Springer*  
*Repository: https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight/*
