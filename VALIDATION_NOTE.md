# Validation note

## Primary design

The report now uses nested stratified five-fold cross-validation as its primary validation design. In each outer iteration, four folds form the development data and the remaining fold is untouched until final testing. Three inner stratified folds select hyperparameters using balanced accuracy. The chosen pipeline is then refitted on the complete outer training data and evaluated once on the outer test fold.

All preprocessing is contained within scikit-learn `Pipeline` and `ColumnTransformer` objects. Imputation, one-hot encoding and scaling are therefore fitted only from the relevant training observations. G3 is used only to define the outcome and is excluded from all predictors.

## Baselines

The simple grade and majority-class baselines are calculated separately on the same five outer test folds as the fitted models. Their reported means and standard deviations are therefore directly comparable with the models.

## Clean execution

A successful execution was completed in a fresh working directory using:

- Python 3.13.5
- pandas 2.2.3
- NumPy 2.3.5
- scikit-learn 1.8.0
- matplotlib 3.10.8
- joblib 1.5.3

Recorded runtime: approximately 1 minute 28 seconds. The console output is preserved in `clean_run.log`; exact package information is recorded in `results/software_environment.json`.

## Report-output checks

The following were checked against the generated output files:

- sample size and pass/fail counts;
- nested outer-test means and standard deviations;
- confusion-matrix counts;
- first-period-only supplementary analysis;
- comparison with Cortez and Silva (2008);
- held-out Random Forest permutation importance;
- figures and table values.

The revised report uses the generated results in `results/`. Probability calibration was not evaluated, and model outputs are not described as validated individual probabilities.
