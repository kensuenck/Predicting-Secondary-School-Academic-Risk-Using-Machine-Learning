# Predicting Secondary-School Academic Risk Using Machine Learning

This folder is the local **v1.0 dissertation release candidate** for Ken Choi Kan Suen's MSc AI and Sustainable Development research report.

## Contents

- `Ken_Suen_Dissertation_Final_Revision_For_Chris.docx` — revised dissertation draft for supervisor feedback.
- `dissertation_analysis.py` — complete analysis workflow.
- `student-mat.csv` — Mathematics component of the public Student Performance dataset.
- `requirements.txt` — recorded Python package versions.
- `results/` — generated tables, figures and software metadata used in the report.
- `clean_run.log` — console output from a successful clean execution in a fresh working directory.
- `VALIDATION_NOTE.md` — validation design, execution and result checks.
- `REVISION_SUMMARY.md` — summary of the eleven completed revision priorities.
- `WORD_COUNT_NOTE.md` — chapter-level word-count check against the dissertation handbook.

## Analysis design

The primary evaluation is nested stratified five-fold cross-validation:

1. Five outer folds estimate final out-of-sample performance.
2. Within each outer training set, three inner stratified folds select hyperparameters using balanced accuracy.
3. Preprocessing and hyperparameter selection are fitted only within training data.
4. Each untouched outer test fold is evaluated once.
5. The five outer-fold predictions are combined for the confusion matrices.

Academic risk is coded as `G3 < 10`, with failure as the positive class. The script evaluates:

- Configuration A: G1 and G2 included.
- Configuration B: G1 included and G2 excluded.
- Configuration C: both G1 and G2 excluded.

Naive baselines are evaluated on the same outer test folds as the fitted models.

## Local execution

From this folder:

```bash
python -m pip install -r requirements.txt
python dissertation_analysis.py --data student-mat.csv --output-dir results_new
```

The script validates that the Mathematics dataset contains 395 observations and writes all outputs into the selected output directory.

## Google Colab

Upload `dissertation_analysis.py`, `student-mat.csv` and `requirements.txt`, then run:

```python
!pip install -q -r requirements.txt
!python dissertation_analysis.py --data student-mat.csv --output-dir results_new
```

## Jupyter Notebook

Keep the script and dataset beside the notebook, then run:

```python
%run dissertation_analysis.py --data student-mat.csv --output-dir results_new
```

## Repository

Project repository:

https://github.com/kensuenck/Predicting-Secondary-School-Academic-Risk-Using-Machine-Learning

After supervisor feedback and final checks, create a GitHub release or tag named `v1.0-dissertation` and ensure it contains the exact version used for submission.

## Important status notes

- The ethics status still requires formal confirmation from the programme or supervisor. The dissertation does not claim approval or exemption that has not been confirmed.
- The local package is a release candidate. It does not establish that the remote GitHub tag already exists.
