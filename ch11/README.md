# Chapter 11: Visualization and Communication

**Book:** _Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches_  
**Publisher:** Springer  
**Repository folder:** `/ch11`  
**Author:** M. Volkan GUNGOR<sup>1</sup>, Assoc. Prof. Dr. Dilek Özdemir GUNGOR<sup>2</sup>  
**Affiliation:** İzmir University of Economics, Vocational School of Health Services<sup>1</sup>, Izmir Katip Celebi University, Faculty of Economics and Administrative Sciences, Department of Data Science and Analytics<sup>2</sup>  
 **Contact:** mvolkang@gmail.com<sup>1</sup>, dilek.ozdemir.gungor@ikcu.edu.tr<sup>2</sup>  
 **ORCID:** https://orcid.org/0000-0001-8093-795X<sup>1</sup>, https://orcid.org/0000-0003-1661-3226<sup>2</sup>

---

## 1. Purpose of This Folder

This folder contains reproducible Python/R scripts supporting Chapter 11: “Visualization and Communication.” These scripts demonstrate the transformation of raw technological data into clear, compelling visual narratives specifically for technology foresight.

The code operationalizes the chapter's discussion on:

- Interactive Foresight Dashboards: Building dynamic interfaces for monitoring technology maturity.
- Foresight-Specific Visuals: Programmatic implementation of Futures Cones, Futures Wheels, and S-Curves.
- Network Visualization: Mapping co-occurrence and co-invention patterns within innovation systems.
- Data Storytelling: Translating technical complexity into strategic intelligence for decision-makers.

The scripts are intended as **chapter companion code** rather than a complete standalone software package.

---

## 2. Repository Contents

The `/ch11` folder should contain the following folders:

```text
ch11/
├── GenAI_Patent_Foresight_Dashboard
└── University_Performance_Dashboard
```

The **`/ch11/GenAI_Patent_Foresight_Dashboard`** folder should contain the following files:

| File                               | Description                                                                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dashboard.py`                     | **Strategic Foresight Hub:** A consolidated dashboard integrating a Roadmap, Trend Radar, Futures Cone, and S-Curve.                                 |
| `futures-cone-GEN-AI.py`           | **Futures Cone Implementation:** Maps scenarios (Projected, Probable, Plausible, Possible) for Generative AI.                                        |
| `generative-AI-futures-wheel.py`   | **Futures Wheel:** A structured version of the wheel mapping STEEPV-based consequences of GenAI adoption for the 2028 horizon.                       |
| `generative-AI-trend-radar.py`     | **Strategic AI Trend Radar:** Visualizes emerging developments across STEEPV categories relative to their maturity horizons (Act, Prepare, Watch).   |
| `s-curve-generative.py`            | **Maturity & S-Curve Analysis:** Uses cumulative patent growth data to identify technology lifecycles and discontinuities.                           |
| `technology-proximity-combined.py` | **Dual Network Analysis:** A comparative visualization tool showing a full technology map alongside a filtered strategic view of CPC co-occurrences. |
| `generative-ai-patent.csv`         | refer to chapter for downloading the dataset                                                                                                         |
| `ch11_requirements.txt`            | Lists the Python dependencies (Streamlit, Plotly, NetworkX, etc.) required to run the scripts.                                                       |
| `README.md`                        | Provides installation, data, and execution instructions for Chapter 11 companion code.                                                               |
| `LICENSE`                          | Recommended license file for the code, if included in the repository.                                                                                |

The **`/ch11/University_Performance_Dashboard`** folder should contain the following files:

| File                                                     | Description                                                                                                                                                                                                                                                                                                            |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `UC1-typesOfDashboards_universityDataSet_userStory.docx` | **Project Requirements & Visualization Guide:** Outlines the goals for university administrators to monitor institutional health, defines specific tasks for data processing, and provides detailed interpretations for each chart type, such as identifying budget inefficiencies or student satisfaction consistency |
| `UC1-typesOfDashboards_universityDataSet.csv`            | **University Performance Data:** A structured dataset containing monthly metrics for Engineering, Business, and Arts faculties, including student enrollment, satisfaction scores, research publications, budget allocations, and attendance rates.                                                                    |
| `UC1-typesOfDashboards_universityDataSet.py`             | **Multi-View Dashboard Application:** A Streamlit-based Python script that builds an interactive web interface to visualize university metrics through five distinct management perspectives: Main Overview, Strategic, Analytical, Operational, and Tactical.                                                         |

Expected folder structure:

```text
ch11/
  ├── University_Performance_Dashboard/
  │   ├── UC1-typesOfDashboards_universityDataSet.csv
  │   ├── UC1-typesOfDashboards_universityDataSet.py
  │   └── UC1-typesOfDashboards_universityDataSet_userStory.docx
  └── GenAI_Patent_Foresight_Dashboard/
      ├── ch11_README.md
      ├── ch11_requirements.txt
      ├── dashboard.py
      ├── futures-cone-GEN-AI.py
      ├── generative-AI-futures-wheel.py
      ├── generative-AI-trend-radar.py
      ├── s-curve-generative-AI.py
      └── technology-proximity-combined.py
```

---

## 3. Data Sources and Licensing

This chapter uses patent data to demonstrate data processing and technology foresight workflows. The repository **does not include raw proprietary data**.

### 3.1 Lens.org Patent Citation Data (GenAI_Patent_Foresight_Dashboard)

The scripts are designed to process CSV exports from Lens.org, focusing on Generative AI patent landscapes.

The `dashboard.py`, `technology-proximity-combined.py` and `s-curve-generative-AI.py` scripts are designed for patent citation exports from **Lens.org**.

- **Data source:** Lens.org
- **Access model:** Institutional subscription / proprietary database. Free access for academic research, subject to Lens.org terms of use
- **Fallback Mechanism:** If the specific local file generative-ai-patent.csv is not found, the scripts utilize synthetic data generation to demonstrate visualization functionality.
- **Redistribution status:** Raw exported patent data are **not redistributed** in this repository.

Recommended Lens.org export fields:

```text
Publication Year
Jurisdiction
Applicants
IPCR Classifications
CPC Classifications
```

### 3.2 University Performance Data Analysis (University_Performance_Dashboard)

The scripts and datasets are designed to process institutional metrics, focusing on multi-level management perspectives within a university setting.

The `UC1-typesOfDashboards_universityDataSet.py` script is designed to visualize performance data across various faculties.

- **Data source:** Institutional Performance Records.
- **Access model:** Provided as a sample CSV dataset (UC1-typesOfDashboards_universityDataSet.csv).
- **Fallback Mechanism:** The script includes a try/except block to handle different CSV delimiters (semicolon or comma) to ensure the data loads correctly.
- **Redistribution status:** The sample dataset is included for educational and demonstration purposes within this environment.
- **Case study used in the chapter:** University Performance One Page Dashboard for Administrators and Faculty Managers.

Recommended export fields:

```text
faculty
month
students
satisfaction
publications
budget
attendance_rate
```

Table content:
| faculty | month | students | satisfaction | publications | budget | attendance_rate |
|-------------|-------|----------|--------------|--------------|--------|------------------|
| Engineering | Jan | 1200 | 82 | 15 | 120000 | 88 |  
| Business | Jan | 800 | 78 | 10 | 90000 | 84 |  
| Arts | Jan | 600 | 85 | 8 | 60000 | 92 |  
| Engineering | Feb | 1220 | 83 | 16 | 118000 | 87 |  
| Business | Feb | 790 | 80 | 9 | 91000 | 85 |  
| Arts | Feb | 610 | 84 | 7 | 61000 | 90 |  
| Engineering | Mar | 1250 | 84 | 17 | 119000 | 89 |  
| Business | Mar | 810 | 79 | 11 | 93000 | 83 |  
| Arts | Mar | 620 | 86 | 9 | 61500 | 91 |

---

## 4. Software Requirements

### 4.1 System Requirements

- Python 3.9 or higher
- Tested with Python 3.14
- 4 GB RAM minimum
- 8 GB RAM recommended for large network graphs

### 4.2 Python Dependencies

The core foresight-tech stack includes:

- **streamlit:** Orchestrates the interactive user interface and dashboards.
- **plotly:** Powers high-fidelity, interactive chart types such as Radars, Cones, and Timelines.
- **networkx:** Facilitates the construction and analysis of network science topologies.
- **pandas** & **numpy:** Essential for manipulating and processing complex foresight datasets.
- **scipy:** Used for mathematical modeling and sigmoid curve fitting.

The required packages are listed in `ch11_requirements.txt`.

Core dependencies:

```text
streamlit>=1.54.0
pandas>=2.3.3
networkx>=3.6.1
plotly>=6.5.2
numpy>=2.4.2
matplotlib>=3.10.9
seaborn>=0.13.2
scipy>=1.17.1
```

Optional packages may be added by users for notebooks, interactive dashboards, or network analysis, but they are not required for the main scripts.

---

## 5. Installation

From the Springer repository root:

```bash
git clone https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight.git
cd applied-quantitative-methods-in-technology-foresight/ch11
pip install -r ch11_requirements.txt
```

Alternatively, create and activate a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

pip install -r ch11_requirements.txt
```

---

## 6. How to Run the Scripts

### 6.1 Launching the Main Hub

To run the primary multi-panel dashboard:

```bash
streamlit run dashboard_v4_exatR.py
```

### 6.2 Running Individual Foresight Tools

To focus on a specific methodology (e.g., the Futures Wheel):

```bash
streamlit run generative-AI-futures-wheel_v1.py
```

---

## 7. Key Foresight Visuals Explained

- **Futures Cone:** A heuristic framework for navigating uncertainty. It expands through time to show Projected (business as usual), Probable (likely), Plausible (could happen), and Possible (might happen) scenarios.
- **Trend Radar:** A navigational tool that situates trends in Act (urgent), Prepare (mid-term momentum), or Watch (early-stage weak signals) zones based on maturity.
- **Futures Wheel:** Designed to map multi-layered, cascading implications of a change agent, from direct Primary impacts to complex systemic Secondary and Tertiary ripples
- **S-Curve (Sigmoid Curve):** Plots performance against time or investment to identify when a technology enters a Saturation phase and is likely to be displaced by a disruptive innovation.
- **Futures Wheel:** A qualitative method for mapping primary and derivative impacts of a trend
- **Network Maps:** Visualize innovation systems by showing interconnected actors (Nodes) and their collaborative relationships (Edges).
- For University Performance Dashboard related story and explanations please refer to ** `UC1-typesOfDashboards_universityDataSet_userStory.docx` **

---

## 8. Suggested Citation

If using the chapter code, please cite the book chapter:

```bibtex
@incollection{Gungor2026Chapter11,
  author    = {Güngör, Mustafa Volkan and Özdemir-Güngör, Dilek},
  title     = {Visualization and Communication},
  booktitle = {Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches},
  chapter   = {11},
  year      = {2026},
  publisher = {Springer}
}
```

If citing the code repository, please use the Springer repository path:

```bibtex
@software{Gungor2026Ch11Code,
  author = {Güngör, Mustafa Volkan and Özdemir-Güngör, Dilek},
  title  = {Chapter 11 Companion Code: Visualization and Communication},
  year   = {2026},
  url    = {https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight/tree/main/ch11}
}
```

---

## 11. License

Unless otherwise stated in the repository, the chapter companion code is provided for academic and instructional use. If a `LICENSE` file is included in `/ch11`, its terms govern reuse, modification, and redistribution of the code.

This repository does not grant redistribution rights for Web of Science, Scopus, Lens.org, or any other third-party proprietary or platform-governed datasets.

---

## 12. Contact

For questions about Chapter 11 code:

**M. Volkan GÜNGÖR**
İzmir University of Economics  
Vocational School of Health Services  
Email: mvolkang@gmail.com

**Dilek Özdemir GÜNGÖR**
İzmir Katip Celebi University  
Department of Data Science and Analytics  
Email: dilek.ozdemir.gungor@ikcu.edu.tr
