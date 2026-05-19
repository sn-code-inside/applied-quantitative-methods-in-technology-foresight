# Chapter 10: AI-Augmented Weak Signal Interpretation for Emerging Technology Foresight

## Authors

Mateus Panizzon, Raquel Janissek-Muniz, Carlos Brito-Cabrera, and Natália Marroni Borges.

## Overview

This folder contains the supplementary educational code for Chapter 10 of *Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches*.

The code demonstrates a simplified, non-proprietary implementation inspired by the IEA-SCALE architecture and the LESCAAI framework. It is intended for teaching, reproducibility, and conceptual understanding of AI-augmented weak signal interpretation. It does **not** reproduce the production IEA-SCALE system.

The scripts illustrate how a small collection of text documents can be organized into a three-level weak-signal intelligence structure:

- **Tower 1:** concise strategic insight;
- **Tower 2:** thematic contexts and intermediate interpretation;
- **Tower 3:** traceable evidence fragments with source metadata.

The educational workflow emphasizes that AI can structure and amplify perception, but weak-signal meaning still depends on human interpretation, validation, and sensemaking.

A related institutional repository from IEA Future Lab is also available at:

https://github.com/IEAfuture/iea-scale-educational

---

## Repository contents

| Path | Description |
|------|-------------|
| `src/pipeline.py` | Main executable script. Loads sample documents, retrieves relevant text chunks, and generates the three-tower outputs. |
| `src/text_io.py` | Loads and parses small `.txt` teaching documents with metadata. |
| `src/retrieval.py` | Provides a lightweight term-frequency retrieval model for educational use. |
| `src/towers.py` | Generates Tower 1, Tower 2, and Tower 3 outputs from retrieved fragments. |
| `data/sample_docs/` | Synthetic, open teaching sample documents used to run the demo without proprietary data. |
| `chapter3_section3_3/` | Standalone educational scripts associated with the multi-agent examples discussed in Section 3.3. |
| `chapter4_code/` | Standalone scripts associated with the Chapter 4 code and prompt implementation examples. |
| `requirements.txt` | Python dependencies required to run the code. |

---

## Software environment and dependencies

Recommended environment:

- Python 3.10 or higher;
- `pip` package manager;
- optional virtual environment.

Install dependencies from inside the `ch10` folder:

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
numpy
pandas
scikit-learn
sentence-transformers
python-dotenv
```

Optional dependencies for API-based demonstrations:

```text
openai
google-generativeai
```

The default pipeline runs offline and does not require paid API keys. Optional LLM-based examples may use external providers if the reader supplies their own credentials.

---

## How to run the scripts

From the repository root:

```bash
cd ch10
pip install -r requirements.txt
python -m src.pipeline --input data/sample_docs --out out
```

Expected outputs:

```text
out/tower1.md
out/tower2.md
out/tower3.json
```

These files reproduce the educational logic presented in the chapter:

1. Load teaching documents;
2. Retrieve relevant fragments;
3. Generate a strategic insight;
4. Generate thematic contexts;
5. Preserve traceable evidence fragments.

The results are intended to reproduce the **logic and structure** of the chapter examples, not to reproduce a production foresight platform.

---

## Step-by-step reproduction guide

1. Clone the Springer repository:

```bash
git clone https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight.git
```

2. Move to the Chapter 10 folder:

```bash
cd applied-quantitative-methods-in-technology-foresight/ch10
```

3. Create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the demo pipeline:

```bash
python -m src.pipeline --input data/sample_docs --out out
```

6. Inspect the generated outputs:

- `out/tower1.md` for the executive-level strategic insight;
- `out/tower2.md` for thematic interpretation;
- `out/tower3.json` for evidence fragments and metadata.

---

## Data sources, license information, and access instructions

The chapter discusses open-access weak-signal material from the European Commission Joint Research Centre (JRC):

**European Commission – Joint Research Centre (JRC)**  
**Weak signals in Science and Technologies (2024)**  
Repository page: https://publications.jrc.ec.europa.eu/repository/handle/JRC140959  
Access date documented for this repository: 13 May 2026.

The JRC repository page provides access to the report and associated publication metadata. Readers can access the open material directly from the JRC repository link above.

The demo code submitted in this folder does **not** include raw proprietary data from Scopus, Web of Science, or PATSTAT. When the chapter refers to publication or patent-based signal detection, readers should consult the original JRC report and reproduce the underlying searches according to their own institutional access rights and database licenses.

The files in `data/sample_docs/` are synthetic teaching examples created only to demonstrate the workflow. They are not extracted from Scopus, Web of Science, PATSTAT, or any restricted database. They are distributed under the same educational scope as this repository.

---

## Notes on proprietary and restricted data

No raw proprietary database exports are included in this folder.

If readers wish to reproduce larger bibliometric or patent-based datasets, they should:

1. Consult the JRC report for the methodological description;
2. Use their own authorized access to databases such as Scopus, Web of Science, PATSTAT, Lens.org, or OpenAlex;
3. Record the exact query, date, filters, and database version used;
4. Respect the license terms of each database.

---

## Known limitations

This code is intentionally simple. It is designed to support graduate-level learning and reproducibility of the chapter logic, not industrial deployment.

Limitations include:

- small synthetic dataset;
- lightweight retrieval model;
- no database backend;
- no scraping infrastructure;
- no production authentication;
- no private prompts or proprietary SCALE implementation details.

---

## License

The educational code and synthetic sample data in this folder are provided under the MIT License, unless otherwise stated by the parent repository.

---

*Part of: Applied Quantitative Methods in Technology Foresight: AI-Enhanced Approaches — Springer*  
*Repository: https://github.com/sn-code-inside/applied-quantitative-methods-in-technology-foresight/*
