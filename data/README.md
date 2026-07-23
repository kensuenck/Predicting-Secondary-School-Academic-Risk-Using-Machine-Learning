# Data

## File

`student-mat.csv` contains the Mathematics component of the Student Performance dataset: 395 students from two Portuguese public secondary schools. The file is semicolon-delimited.

## Outcome used in this project

- `G3`: final Mathematics grade on a 0–20 scale.
- `academic_risk = 1` when `G3 < 10`; otherwise `academic_risk = 0`.
- `G3` is used only to construct the outcome and is excluded from all predictors.

## Predictor groups

The dataset contains:

- academic variables, including `G1`, `G2`, absences, study time and previous failures;
- demographic and family variables, including age, sex, parental education and parental occupations;
- school, social and behavioural questionnaire variables, including support, free time, going out, health and alcohol-use measures.

Not every questionnaire variable would necessarily be routinely available or appropriate for operational use in a school. The timing of several questionnaire variables is not documented precisely enough to support a strong start-of-year prediction claim.

## Attribution

Cortez, P. and Silva, A. (2008) ‘Using data mining to predict secondary school student performance’, in Brito, A. and Teixeira, J. (eds.) *Proceedings of the 5th Annual Future Business Technology Conference*. Porto: EUROSIS, pp. 5–12.

Original dataset source: UCI Machine Learning Repository, Student Performance dataset.

## Responsible use

Although the data are public and pseudonymised, they concern young people and include sensitive self-reports. Do not use the dataset or models to make consequential decisions about real students without appropriate legal, ethical, governance and stakeholder review.
