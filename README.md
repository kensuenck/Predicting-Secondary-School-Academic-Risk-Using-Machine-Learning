# Predicting Secondary-School Academic Risk Using Machine Learning

Reproducibility materials for the MSc AI and Sustainable Development dissertation by **Ken Choi Kan Suen**, School of Government, University of Birmingham (2026).

## Research question

> To what extent can machine-learning models predict academic risk among secondary-school students using existing academic, demographic and social variables?

Academic risk is defined as a final Mathematics grade below 10 on a 0–20 scale. The analysis compares Logistic Regression, Decision Tree and Random Forest models under three predictor configurations:

- **A — G1 and G2:** first- and second-period grades included.
- **B — G1 only:** first-period grade included; second-period grade excluded.
- **C — no period grades:** both G1 and G2 excluded.

## Main finding

Prediction is strong after earlier Mathematics grades are available, but a simple rule based on the second-period grade performs similarly to the machine-learning models. Without earlier grades, performance declines substantially and the models identify only around half of the students who fail. The results support cautious, low-stakes professional review rather than automatic labelling or allocation of support.

## Repository contents

```text
.
├── README.md
├── CITATION.cff
├── CHANGELOG.md
├── requirements.txt
├── environment.yml
├── src/
│   └── dissertation_analysis.py
├── data/
│   ├── student-mat.csv
│   └── README.md
├── notebooks/
│   └── run_analysis.ipynb
├── results/
│   ├── *.csv
│   ├── software_environment.json
│   ├── OUTPUT_MANIFEST.txt
│   ├── README.md
│   └── figures/
├── reproducibility/
│   ├── CLEAN_RUN.log
│   ├── VALIDATION.md
│   └── SHA256SUMS.txt
└── docs/
    ├── Ken_Suen_Dissertation_Submission_Ready.docx
    ├── Ken_Suen_Dissertation_Submission_Ready.pdf
    └── ABSTRACT.md
```

## Method

The primary evaluation uses **nested stratified five-fold cross-validation**:

1. Five outer folds estimate final out-of-sample performance.
2. Within each outer training set, three inner stratified folds select hyperparameters using balanced accuracy.
3. Imputation, one-hot encoding and scaling are fitted only within the relevant training data through scikit-learn pipelines.
4. Each untouched outer test fold is evaluated once.
5. Naive baselines are evaluated on the same outer folds as the fitted models.
6. Random Forest permutation importance is calculated only on held-out outer-test observations.

The final grade (`G3`) is used only to construct the target and is excluded from every predictor set.

## Quick start

### Local Python

Python 3.13 was used for the archived run.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the exact recorded dependencies and run the analysis:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python src/dissertation_analysis.py \
  --data data/student-mat.csv \
  --output-dir results_local
```

The command writes all tables, figures and environment metadata to `results_local/`. The tracked `results/` directory contains the archived outputs used in the dissertation.

### Conda

```bash
conda env create -f environment.yml
conda activate academic-risk-dissertation
python src/dissertation_analysis.py --data data/student-mat.csv --output-dir results_local
```

### Google Colab or Jupyter

Open [`notebooks/run_analysis.ipynb`](notebooks/run_analysis.ipynb). The notebook contains cells to install dependencies, run the script, inspect the principal tables and display the figures.

## Key archived outputs

- `results/nested_cv_summary_wide.csv` — mean and standard deviation of performance measures.
- `results/confusion_matrices.csv` — combined held-out predictions for each model and configuration.
- `results/original_study_comparison.csv` — comparison with Cortez and Silva (2008).
- `results/nested_cv_selected_hyperparameters.csv` — settings selected within the inner folds.
- `results/permutation_importance_summary.csv` — held-out Random Forest permutation importance.
- `results/descriptive_statistics.csv` — selected sample characteristics.
- `results/figures/` — figures used in the report.

## Data source and attribution

The repository contains the Mathematics component of the UCI Student Performance dataset. The data were originally described in:

> Cortez, P. and Silva, A. (2008) ‘Using data mining to predict secondary school student performance’, in Brito, A. and Teixeira, J. (eds.) *Proceedings of the 5th Annual Future Business Technology Conference*. Porto: EUROSIS, pp. 5–12.

See [`data/README.md`](data/README.md) for variables, scope and responsible-use notes.

## Reproducibility

The archived analysis was run in a fresh working directory. Exact software versions are recorded in `results/software_environment.json`, the console log is stored in `reproducibility/CLEAN_RUN.log`, and file checksums are listed in `reproducibility/SHA256SUMS.txt`.

The dissertation PDF and DOCX correspond to the materials in this package. Probability calibration was not evaluated; model outputs should therefore not be interpreted as validated individual probabilities.

## Ethical and practical scope

The analysis uses an existing public, pseudonymised secondary dataset and involves no new participant recruitment or data collection. The dataset nevertheless includes sensitive self-reported variables. These materials are provided for academic research and reproducibility, not for operational scoring of students. Any practical application would require contemporary local data, institutional ethics and data-governance review, external validation, stakeholder involvement and a contestable human-review process.

## Citation

Use the citation information in [`CITATION.cff`](CITATION.cff). A plain-text form is:

> Suen, K.C.K. (2026) *Predicting Secondary-School Academic Risk Using Machine Learning*. Version 1.0.0. GitHub repository.

## Repository URL

https://github.com/kensuenck/Predicting-Secondary-School-Academic-Risk-Using-Machine-Learning
