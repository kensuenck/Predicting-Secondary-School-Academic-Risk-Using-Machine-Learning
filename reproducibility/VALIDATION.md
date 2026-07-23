# Reproducibility validation

## Validation design

The primary analysis uses nested stratified five-fold cross-validation. In each outer iteration, four folds form the development data and the remaining fold is untouched until final testing. Three inner stratified folds select hyperparameters using balanced accuracy. The selected pipeline is refitted on the full outer-training data and evaluated once on the outer-test fold.

All imputation, one-hot encoding and scaling are contained within scikit-learn `Pipeline` and `ColumnTransformer` objects. The final grade (`G3`) is used only to construct the outcome and is excluded from all predictors.

## Baselines

The grade-rule and majority-class baselines are evaluated separately on the same outer test folds as the fitted models. Their means and standard deviations are therefore directly comparable with the model estimates.

## Archived run

The release package records:

- Python and package versions in `results/software_environment.json`;
- console output in `reproducibility/CLEAN_RUN.log`;
- generated-file inventory in `results/OUTPUT_MANIFEST.txt`;
- SHA-256 hashes in `reproducibility/SHA256SUMS.txt`.

## Report-output checks

The following were checked against generated files:

- sample size and pass/fail counts;
- outer-test means and standard deviations;
- confusion-matrix counts;
- all three feature configurations;
- comparison with Cortez and Silva (2008);
- held-out Random Forest permutation importance;
- figures and table values.

Probability calibration was not evaluated. Scores must not be interpreted as validated estimates of an individual student's absolute probability of failure.
