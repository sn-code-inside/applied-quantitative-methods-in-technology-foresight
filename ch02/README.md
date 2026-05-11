# Chapter 2: Fundamentals of Data in Technology Foresight

**Book:** *Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches*  
**Publisher:** Springer  
**Repository folder:** `/ch02`  
**Author:** Asst. Prof. Dr. Kemal Yayla  
**Affiliation:** Izmir Katip Celebi University, Faculty of Economics and Administrative Sciences, Department of Data Science and Analytics  
**Contact:** kemal.yayla@ikcu.edu.tr  
**ORCID:** https://orcid.org/0000-0001-9064-611X  

---

## 1. Purpose of This Folder

This folder contains the reproducible Python code supporting **Chapter 2: “Fundamentals of Data in Technology Foresight.”** The scripts demonstrate how bibliometric and patent data can be processed, structured, analyzed, and visualized for technology foresight applications.

The code accompanies the chapter’s applied discussion of:

- Web of Science plain-text data conversion into structured JSON format;
- data preparation and quality-aware bibliometric workflows;
- patent-based science–technology linkage analysis;
- Sleeping Beauty impact analysis using patent citations;
- temporal, geographic, and technological diffusion visualization.

The scripts are intended as **chapter companion code** rather than a complete standalone software package.

---

## 2. Repository Contents

The `/ch02` folder should contain the following files:

| File | Description |
|---|---|
| `wos2json.py` | Converts Web of Science plain-text exports into structured JSON records. It supports batch processing, multi-line field parsing, character encoding handling, and standardized bibliographic field mapping. |
| `SB_Impact_Analyzer.py` | Analyzes the technology-side impact of Sleeping Beauty publications using Lens.org patent citation metadata. It computes awakening phases, diffusion patterns, science–technology coupling indicators, and maturity trajectories. |
| `SB_Impact_Visualization.py` | Creates heliocentric and temporal visualizations of technology-domain evolution, geographic diffusion, and patent citation dynamics from Lens.org patent export files. |
| `requirements.txt` | Lists the Python dependencies required to run the scripts. |
| `README.md` | Provides installation, data, and execution instructions for Chapter 2 companion code. |
| `LICENSE` | Recommended license file for the code, if included in the repository. |

Expected folder structure:

```text
ch02/
├── README.md
├── requirements.txt
├── wos2json.py
├── SB_Impact_Analyzer.py
├── SB_Impact_Visualization.py
└── LICENSE
```

---

## 3. Data Sources and Licensing

This chapter uses bibliometric and patent data to demonstrate data processing and technology foresight workflows. The repository **does not include raw proprietary data**.

### 3.1 Web of Science Data

The `wos2json.py` script is designed for **Web of Science Core Collection** plain-text exports.

- **Data source:** Web of Science Core Collection
- **Access model:** Institutional subscription / proprietary database
- **Redistribution status:** Raw records are **not redistributed** in this repository due to licensing restrictions.
- **Extraction date:** December 2025, approximately five months before the Springer repository submission deadline.
- **Example topic used in the chapter:** Nursing simulation technologies, including virtual reality, augmented reality, metaverse, haptics, nursing, simulation, and training-related publications.

Example search query used for the nursing simulation case:

```text
TS=(("virtual reality" OR "augmented reality" OR metaverse OR haptic*) 
AND (nursing OR nurs*) 
AND (simulat* OR train*))
```

Recommended Web of Science export settings:

```text
Database: Web of Science Core Collection
Document type: Article and related scholarly publication records, depending on chapter scope
Publication years: 2022–2024
Export format: Plain Text File
Record content: Full Record and Cited References
File extension: .txt
```

Readers with institutional access can reproduce the dataset by executing the same search and exporting records using the settings above.

### 3.2 Lens.org Patent Citation Data

The `SB_Impact_Analyzer.py` and `SB_Impact_Visualization.py` scripts are designed for patent citation exports from **Lens.org**.

- **Data source:** Lens.org
- **Access model:** Free access for academic research, subject to Lens.org terms of use
- **Redistribution status:** Raw exported patent data are **not redistributed** in this repository.
- **Extraction date:** December 2025, approximately five months before the Springer repository submission deadline.
- **Case study used in the chapter:** Patent citations to Folkman’s 1971 angiogenesis discovery.

Recommended Lens.org export fields:

```text
Publication Year
Publication Date
Application Date
Earliest Priority Date
Jurisdiction
Title
CPC Classifications
Cites Patent Count
Cited by Patent Count
NPL Citation Count
Simple Family Size
Extended Family Size
```

Suggested reproduction procedure:

1. Access Lens.org.
2. Search for the focal scientific publication.
3. Use the “Cited by Patents” functionality.
4. Export citing patent records as CSV.
5. Include publication year, jurisdiction, CPC classifications, citation counts, and patent family indicators.
6. Save the exported file as, for example, `folkman_patent_citations.csv`.

### 3.3 Sample Data

No raw Web of Science, Scopus, or proprietary database records are included in this repository.

If users want to test the scripts without proprietary data, they may create a small synthetic CSV file with the same column names listed above for the patent analysis scripts, or use their own institutionally obtained exports.

---

## 4. Software Requirements

### 4.1 System Requirements

- Python 3.8 or higher
- Tested with Python 3.11
- 4 GB RAM minimum
- 8 GB RAM recommended for larger bibliometric or patent datasets

### 4.2 Python Dependencies

The required packages are listed in `requirements.txt`.

Core dependencies:

```text
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
pycountry>=20.7.3
```

Optional packages may be added by users for notebooks, interactive dashboards, or network analysis, but they are not required for the main scripts.

---

## 5. Installation

From the Springer repository root:

```bash
git clone https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight.git
cd applied-quantitative-methods-in-technology-foresight/ch02
pip install -r requirements.txt
```

Alternatively, create and activate a virtual environment before installing dependencies:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate      # Windows PowerShell

pip install -r requirements.txt
```

---

## 6. How to Run the Scripts

### 6.1 Convert Web of Science Plain-Text Exports to JSON

Place one or more Web of Science `.txt` export files in the same folder as `wos2json.py`, or provide a path to the folder containing the exports.

Interactive mode:

```bash
python wos2json.py
```

Automatic mode:

```bash
python wos2json.py --auto
```

Programmatic usage:

```python
from wos2json import convert_wos_to_json

files = ["savedrecs1.txt", "savedrecs2.txt"]

output = convert_wos_to_json(
    input_files=files,
    output_file="wos_articles.json",
    pretty_print=True,
    include_original=False
)

print(output)
```

Expected output:

```text
wos_articles.json
```

The resulting JSON file contains standardized bibliographic fields such as title, journal name, publication year, DOI, authors, abstract, and Web of Science ID.

---

### 6.2 Analyze Patent-Based Sleeping Beauty Impact

Use `SB_Impact_Analyzer.py` with a Lens.org patent citation CSV export.

Example:

```python
from SB_Impact_Analyzer import analyze_patent_sleeping_beauty

results = analyze_patent_sleeping_beauty(
    csv_file_path="folkman_patent_citations.csv",
    discovery_year=1971,
    period_length=5,
    save_figures=True
)

if results:
    print(results["report"])
```

Expected outputs may include:

```text
sleeping_beauty_impact_analysis.png
sleeping_beauty_impact_report.txt
```

The script estimates technology-side diffusion phases based on cumulative patent citation uptake. These phases should be interpreted as **patent-based technological diffusion phases**, not as direct scientific citation awakening phases.

---

### 6.3 Create Heliocentric Technology and Geographic Visualizations

Use `SB_Impact_Visualization.py` with a Lens.org patent citation CSV export.

Example:

```python
from SB_Impact_Visualization import LensPatentAnalyzer

analyzer = LensPatentAnalyzer("folkman_patent_citations.csv")
results = analyzer.run_complete_analysis(save_figures=True)

print(results["report"])
```

Expected outputs may include:

```text
technology_heliocentric.png
geographic_heliocentric.png
temporal_trends.png
analysis_report.txt
```

The visualizations show how patent-based technological applications radiate from a focal scientific discovery across time, technology domains, and jurisdictions.

---

## 7. Code–Chapter Mapping

| Chapter Component | Script | Function in the Chapter |
|---|---|---|
| Data types and bibliometric data structures | `wos2json.py` | Demonstrates how semi-structured Web of Science text exports can be converted into structured JSON data. |
| Data quality and preprocessing | `wos2json.py` | Shows practical issues in encoding, multi-line fields, missing values, and standardized field mapping. |
| Science–technology linkage | `SB_Impact_Analyzer.py` | Uses patent citations to operationalize the movement from scientific discovery to technological application. |
| Sleeping Beauty case study | `SB_Impact_Analyzer.py` | Calculates dormancy, awakening, widespread diffusion, and maturity indicators. |
| Visual analytics in technology foresight | `SB_Impact_Visualization.py` | Produces heliocentric and temporal visualizations for technology-domain and geographic diffusion. |

---

## 8. Notes on Reproducibility

Because the original Web of Science and Lens.org exports are not redistributed, exact numerical replication requires users to obtain equivalent data through their own institutional or platform access.

To improve reproducibility, this README provides:

- the data sources used;
- the approximate extraction period;
- the search query and export parameters;
- the expected file formats;
- required column names;
- dependency information;
- execution examples.

Small differences in search results may occur if databases update records, citations, metadata, or indexing coverage after December 2025.

---

## 9. Known Limitations

1. The Web of Science script assumes standard plain-text export formatting.
2. Lens.org CSV column names may vary depending on export settings; users should rename columns to match the expected names if necessary.
3. Patent-based awakening phases are operational proxies for technology-side diffusion and should not be interpreted as direct measures of scientific recognition.
4. CPC classification parsing depends on the structure of the exported `CPC Classifications` field.
5. The scripts are designed for pedagogical and reproducible research demonstration purposes; users should validate outputs before using them for policy or strategic decision-making.

---

## 10. Suggested Citation

If using the chapter code, please cite the book chapter:

```bibtex
@incollection{Yayla2026Chapter2,
  author    = {Yayla, Kemal},
  title     = {Fundamentals of Data in Technology Foresight},
  booktitle = {Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches},
  chapter   = {2},
  year      = {2026},
  publisher = {Springer}
}
```

If citing the code repository, please use the Springer repository path:

```bibtex
@software{Yayla2026Ch02Code,
  author = {Yayla, Kemal},
  title  = {Chapter 2 Companion Code: Fundamentals of Data in Technology Foresight},
  year   = {2026},
  url    = {https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight/tree/main/ch02}
}
```

---

## 11. License

Unless otherwise stated in the repository, the chapter companion code is provided for academic and instructional use. If a `LICENSE` file is included in `/ch02`, its terms govern reuse, modification, and redistribution of the code.

This repository does not grant redistribution rights for Web of Science, Scopus, Lens.org, or any other third-party proprietary or platform-governed datasets.

---

## 12. Contact

For questions about Chapter 2 code:

**Kemal Yayla**  
Izmir Katip Celebi University  
Department of Data Science and Analytics  
Email: kemal.yayla@ikcu.edu.tr
