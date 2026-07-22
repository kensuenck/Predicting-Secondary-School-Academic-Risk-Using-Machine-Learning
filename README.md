# Dissertation analysis: secondary-school academic risk

This package reproduces the quantitative analysis reported in the dissertation **Predicting Secondary-School Academic Risk Using Machine Learning**.

## Research design

- Dataset: `student-mat.csv` (395 Mathematics students; semicolon-delimited).
- Positive class: academic risk/failure, defined as `G3 < 10`.
- Models: Logistic Regression, Decision Tree and Random Forest.
- Feature configurations:
  - A: first- and second-period grades included;
  - B: first-period grade included, second-period grade excluded;
  - C: both prior grades excluded.
- Primary evaluation: repeated stratified five-fold cross-validation with five repetitions.
- Tuning sensitivity analysis: nested stratified cross-validation, with inner folds selecting hyperparameters by balanced accuracy and outer folds estimating final performance.
- Leakage controls: preprocessing is inside each scikit-learn pipeline; G3 is never a predictor; outer test folds do not influence preprocessing or tuning.

## Installation

A clean virtual environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Recommended reproducibility workflow

Run the primary analysis first:

```bash
python dissertation_analysis.py --data student-mat.csv --output-dir results
```

Then run the computationally slower nested-tuning sensitivity analysis into the same output folder:

```bash
python dissertation_analysis.py --data student-mat.csv --output-dir results --nested-only
```

Together, these commands reproduce the tables, figures and sensitivity results supplied in `results/`. The nested analysis is separated only for convenience and runtime; it uses untouched outer test folds and inner validation folds exactly as described in the dissertation.


## Outputs

The analysis produces:

- repeated cross-validation fold-level and summary results;
- out-of-fold confusion matrices;
- naïve-baseline and original-study comparisons;
- nested cross-validation sensitivity results and selected hyperparameters;
- held-out permutation importance;
- Logistic Regression coefficients;
- descriptive statistics and missing-data checks;
- all figures used in the revised dissertation;
- exact software-version metadata.

Precomputed outputs used in the attached draft are supplied in `results/` so that every reported table can be checked directly.

## Interpretation note

The script estimates predictive performance, not causal effects. Feature importance and coefficients should not be interpreted as evidence that changing a variable will cause a student's result to improve.
