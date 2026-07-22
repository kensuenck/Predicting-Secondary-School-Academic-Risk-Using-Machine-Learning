#!/usr/bin/env python3
"""Reproduce the dissertation analysis on secondary-school academic risk.

The script performs the following steps in one execution:
1. loads and validates the public Mathematics dataset;
2. defines academic risk as G3 < 10 (risk/failure is the positive class);
3. evaluates Logistic Regression, Decision Tree and Random Forest models;
4. uses leakage-controlled preprocessing pipelines;
5. reports repeated stratified cross-validation results;
6. runs nested cross-validation as a hyperparameter-tuning sensitivity analysis;
7. produces out-of-fold confusion matrices, held-out permutation importance,
   descriptive statistics, baselines, tables and figures.

All outputs are written to the directory supplied through --output-dir.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


RANDOM_SEED = 42
POSITIVE_LABEL = 1  # 1 = academic risk / final Mathematics failure


@dataclass(frozen=True)
class FeatureConfiguration:
    code: str
    label: str
    excluded_columns: tuple[str, ...]
    naive_source_grade: str | None


CONFIGURATIONS = (
    FeatureConfiguration(
        code="A",
        label="With first- and second-period grades",
        excluded_columns=(),
        naive_source_grade="G2",
    ),
    FeatureConfiguration(
        code="B",
        label="With first-period grade only",
        excluded_columns=("G2",),
        naive_source_grade="G1",
    ),
    FeatureConfiguration(
        code="C",
        label="Without first- or second-period grades",
        excluded_columns=("G1", "G2"),
        naive_source_grade=None,
    ),
)

MAIN_CONFIGURATION_CODES = {"A", "C"}


METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "precision_risk",
    "recall_risk",
    "f1_risk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete dissertation modelling workflow."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("student-mat.csv"),
        help="Path to the semicolon-delimited student-mat.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for all tables, figures and metadata.",
    )
    parser.add_argument(
        "--nested-only",
        action="store_true",
        help="Run only the nested cross-validation sensitivity analysis. By default the script runs the primary analysis.",
    )
    return parser.parse_args()


def load_and_validate_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path.resolve()}. "
            "Place student-mat.csv beside the script or pass --data."
        )
    df = pd.read_csv(path, sep=";")
    required = {"G1", "G2", "G3", "school", "sex", "age", "failures", "absences"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if len(df) != 395:
        raise ValueError(
            f"Expected 395 Mathematics records, found {len(df)}. "
            "Check that the correct dataset was supplied."
        )
    return df


def prepare_xy(
    df: pd.DataFrame, configuration: FeatureConfiguration
) -> tuple[pd.DataFrame, pd.Series]:
    y = (df["G3"] < 10).astype(int).rename("academic_risk")
    drop_columns = ["G3", *configuration.excluded_columns]
    X = df.drop(columns=drop_columns).copy()
    return X, y


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns, categorical_columns = split_columns(X)
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Return fixed, pre-specified models for the primary repeated CV analysis.

    These complexity-constrained settings are specified before evaluation. They are
    not selected from the reported outer test results. Nested cross-validation below
    assesses sensitivity to a limited set of alternative settings.
    """
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", build_preprocessor(X, scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocess", build_preprocessor(X, scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        max_depth=5,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=RANDOM_SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def scoring_dictionary() -> dict[str, Any]:
    from sklearn.metrics import make_scorer

    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_risk": make_scorer(
            precision_score, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "recall_risk": make_scorer(
            recall_score, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "f1_risk": make_scorer(
            f1_score, pos_label=POSITIVE_LABEL, zero_division=0
        ),
    }


def calculate_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_risk": precision_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "recall_risk": recall_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
        "f1_risk": f1_score(
            y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0
        ),
    }


def naive_predictions(df: pd.DataFrame, config: FeatureConfiguration) -> np.ndarray:
    if config.naive_source_grade is None:
        # The majority class is pass (risk = 0).
        return np.zeros(len(df), dtype=int)
    # Original-study baseline: use the latest available period grade directly.
    return (df[config.naive_source_grade] < 10).astype(int).to_numpy()


def repeated_cv_analysis(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    repeated_cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=RANDOM_SEED
    )
    oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for config in CONFIGURATIONS:
        X, y = prepare_xy(df, config)
        models = build_models(X)

        baseline_pred = naive_predictions(df, config)
        baseline_metrics = calculate_metrics(y, baseline_pred)
        for metric, value in baseline_metrics.items():
            summary_rows.append(
                {
                    "configuration": config.code,
                    "configuration_label": config.label,
                    "model": "Naive baseline",
                    "metric": metric,
                    "mean": value,
                    "standard_deviation": 0.0,
                    "evaluations": 1,
                }
            )
        tn, fp, fn, tp = confusion_matrix(y, baseline_pred, labels=[0, 1]).ravel()
        confusion_rows.append(
            {
                "configuration": config.code,
                "configuration_label": config.label,
                "model": "Naive baseline",
                "true_pass": int(tn),
                "false_risk_warning": int(fp),
                "missed_failure": int(fn),
                "correctly_identified_failure": int(tp),
            }
        )

        for model_name, model in models.items():
            cv_result = cross_validate(
                model,
                X,
                y,
                cv=repeated_cv,
                scoring=scoring_dictionary(),
                n_jobs=-1,
                return_train_score=False,
                error_score="raise",
            )
            for fold_index in range(len(cv_result["test_accuracy"])):
                row: dict[str, Any] = {
                    "configuration": config.code,
                    "configuration_label": config.label,
                    "model": model_name,
                    "evaluation": fold_index + 1,
                }
                for metric in METRIC_NAMES:
                    row[metric] = float(cv_result[f"test_{metric}"][fold_index])
                fold_rows.append(row)

            for metric in METRIC_NAMES:
                values = np.asarray(cv_result[f"test_{metric}"], dtype=float)
                summary_rows.append(
                    {
                        "configuration": config.code,
                        "configuration_label": config.label,
                        "model": model_name,
                        "metric": metric,
                        "mean": values.mean(),
                        "standard_deviation": values.std(ddof=1),
                        "evaluations": len(values),
                    }
                )

            oof_pred = cross_val_predict(
                model,
                X,
                y,
                cv=oof_cv,
                method="predict",
                n_jobs=-1,
            )
            tn, fp, fn, tp = confusion_matrix(y, oof_pred, labels=[0, 1]).ravel()
            confusion_rows.append(
                {
                    "configuration": config.code,
                    "configuration_label": config.label,
                    "model": model_name,
                    "true_pass": int(tn),
                    "false_risk_warning": int(fp),
                    "missed_failure": int(fn),
                    "correctly_identified_failure": int(tp),
                }
            )

    return (
        pd.DataFrame(fold_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(confusion_rows),
    )


def nested_parameter_grids() -> dict[str, dict[str, list[Any]]]:
    """Limited, theory-informed grids used only inside inner validation folds."""
    return {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "Decision Tree": {
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [3, 5],
        },
        "Random Forest": {
            "model__n_estimators": [100],
            "model__max_depth": [3, 5, None],
            "model__min_samples_leaf": [3, 5],
        },
    }


def nested_cv_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested CV sensitivity analysis for the two main feature configurations.

    Outer test folds estimate performance. Inner validation folds select
    hyperparameters using balanced accuracy. The outer test fold never influences
    preprocessing or selection.
    """
    result_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    grids = nested_parameter_grids()

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for config in CONFIGURATIONS:
        if config.code not in MAIN_CONFIGURATION_CODES:
            continue
        X, y = prepare_xy(df, config)
        models = build_models(X)

        for model_name, base_pipeline in models.items():
            for outer_fold, (train_index, test_index) in enumerate(
                outer_cv.split(X, y), start=1
            ):
                X_train = X.iloc[train_index]
                X_test = X.iloc[test_index]
                y_train = y.iloc[train_index]
                y_test = y.iloc[test_index]

                inner_cv = StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=RANDOM_SEED + outer_fold,
                )
                search = GridSearchCV(
                    estimator=clone(base_pipeline),
                    param_grid=grids[model_name],
                    scoring="balanced_accuracy",
                    cv=inner_cv,
                    n_jobs=1,
                    refit=True,
                    error_score="raise",
                    return_train_score=False,
                )
                search.fit(X_train, y_train)
                prediction = search.predict(X_test)
                metrics = calculate_metrics(y_test, prediction)
                result_rows.append(
                    {
                        "configuration": config.code,
                        "configuration_label": config.label,
                        "model": model_name,
                        "outer_fold": outer_fold,
                        **metrics,
                    }
                )
                parameter_rows.append(
                    {
                        "configuration": config.code,
                        "configuration_label": config.label,
                        "model": model_name,
                        "outer_fold": outer_fold,
                        "inner_best_balanced_accuracy": search.best_score_,
                        "best_parameters": json.dumps(
                            search.best_params_, sort_keys=True
                        ),
                    }
                )

    nested_folds = pd.DataFrame(result_rows)
    summary_rows: list[dict[str, Any]] = []
    for (config, label, model), group in nested_folds.groupby(
        ["configuration", "configuration_label", "model"], sort=False
    ):
        for metric in METRIC_NAMES:
            summary_rows.append(
                {
                    "configuration": config,
                    "configuration_label": label,
                    "model": model,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "standard_deviation": group[metric].std(ddof=1),
                    "outer_folds": len(group),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(parameter_rows)


def permutation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for config in CONFIGURATIONS:
        if config.code not in MAIN_CONFIGURATION_CODES:
            continue
        X, y = prepare_xy(df, config)
        model = build_models(X)["Random Forest"]
        for fold, (train_index, test_index) in enumerate(outer_cv.split(X, y), start=1):
            fitted = clone(model).fit(X.iloc[train_index], y.iloc[train_index])
            importance = permutation_importance(
                fitted,
                X.iloc[test_index],
                y.iloc[test_index],
                scoring="balanced_accuracy",
                n_repeats=10,
                random_state=RANDOM_SEED + fold,
                n_jobs=1,
            )
            for feature_index, feature in enumerate(X.columns):
                for repetition, value in enumerate(
                    importance.importances[feature_index], start=1
                ):
                    rows.append(
                        {
                            "configuration": config.code,
                            "configuration_label": config.label,
                            "outer_fold": fold,
                            "repetition": repetition,
                            "feature": feature,
                            "importance": float(value),
                        }
                    )
    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["configuration", "configuration_label", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            standard_deviation=("importance", "std"),
            observations=("importance", "size"),
        )
        .sort_values(["configuration", "mean_importance"], ascending=[True, False])
    )
    return summary


def logistic_coefficients(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
        if config.code not in MAIN_CONFIGURATION_CODES:
            continue
        X, y = prepare_xy(df, config)
        pipeline = build_models(X)["Logistic Regression"]
        pipeline.fit(X, y)
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        coefficients = pipeline.named_steps["model"].coef_[0]
        for feature, coefficient in zip(feature_names, coefficients, strict=True):
            rows.append(
                {
                    "configuration": config.code,
                    "configuration_label": config.label,
                    "encoded_feature": feature,
                    "coefficient": float(coefficient),
                    "absolute_coefficient": abs(float(coefficient)),
                    "direction": "higher predicted risk" if coefficient > 0 else "lower predicted risk",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["configuration", "absolute_coefficient"], ascending=[True, False]
    )


def descriptive_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk = (df["G3"] < 10).astype(int)
    labelled = df.assign(academic_risk=risk)
    groups = {
        "All students": labelled,
        "Passing students": labelled[labelled["academic_risk"] == 0],
        "Failing students": labelled[labelled["academic_risk"] == 1],
    }

    rows: list[dict[str, Any]] = []

    def add_continuous(name: str, column: str, use_median: bool = False) -> None:
        row: dict[str, Any] = {"characteristic": name}
        for group_name, group in groups.items():
            values = group[column]
            if use_median:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                row[group_name] = f"{values.median():.1f} [{q1:.1f}, {q3:.1f}]"
            else:
                row[group_name] = f"{values.mean():.2f} ({values.std(ddof=1):.2f})"
        rows.append(row)

    def add_binary(name: str, column: str, positive: Any) -> None:
        row = {"characteristic": name}
        for group_name, group in groups.items():
            count = int((group[column] == positive).sum())
            percent = 100 * count / len(group)
            row[group_name] = f"{count} ({percent:.1f}%)"
        rows.append(row)

    rows.append(
        {
            "characteristic": "Students, n",
            **{name: str(len(group)) for name, group in groups.items()},
        }
    )
    add_binary("Female, n (%)", "sex", "F")
    add_binary("Gabriel Pereira school, n (%)", "school", "GP")
    add_binary("Urban home address, n (%)", "address", "U")
    add_continuous("Age, mean (SD)", "age")
    add_continuous("Mother's education, mean (SD)", "Medu")
    add_continuous("Father's education, mean (SD)", "Fedu")
    add_continuous("Study time, mean (SD)", "studytime")
    add_continuous("Previous class failures, mean (SD)", "failures")
    add_continuous("Absences, median [IQR]", "absences", use_median=True)
    add_continuous("First-period grade (G1), mean (SD)", "G1")
    add_continuous("Second-period grade (G2), mean (SD)", "G2")
    add_continuous("Final grade (G3), mean (SD)", "G3")
    add_binary("Family educational support, n (%)", "famsup", "yes")
    add_binary("School educational support, n (%)", "schoolsup", "yes")
    add_binary("Internet access at home, n (%)", "internet", "yes")
    add_continuous("Going out with friends, mean (SD)", "goout")

    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_values": [int(df[column].isna().sum()) for column in df.columns],
        }
    )
    return pd.DataFrame(rows), missing


def original_study_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    original = pd.DataFrame(
        [
            {"configuration": "A", "model": "Naive baseline", "original_accuracy": 0.919},
            {"configuration": "A", "model": "Decision Tree", "original_accuracy": 0.907},
            {"configuration": "A", "model": "Random Forest", "original_accuracy": 0.912},
            {"configuration": "B", "model": "Naive baseline", "original_accuracy": 0.838},
            {"configuration": "B", "model": "Decision Tree", "original_accuracy": 0.831},
            {"configuration": "B", "model": "Random Forest", "original_accuracy": 0.830},
            {"configuration": "C", "model": "Naive baseline", "original_accuracy": 0.671},
            {"configuration": "C", "model": "Decision Tree", "original_accuracy": 0.653},
            {"configuration": "C", "model": "Random Forest", "original_accuracy": 0.705},
        ]
    )
    current = summary[summary["metric"] == "accuracy"].copy()
    current = current[current["model"].isin(["Naive baseline", "Decision Tree", "Random Forest"])]
    current = current.rename(
        columns={"mean": "current_accuracy", "standard_deviation": "current_standard_deviation"}
    )[
        [
            "configuration",
            "configuration_label",
            "model",
            "current_accuracy",
            "current_standard_deviation",
        ]
    ]
    result = original.merge(current, on=["configuration", "model"], how="left")
    result["difference_current_minus_original"] = (
        result["current_accuracy"] - result["original_accuracy"]
    )
    return result


def create_figures(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    risk_counts = (df["G3"] < 10).value_counts().sort_index()
    labels = ["Pass", "Fail"]
    values = [int(risk_counts.get(False, 0)), int(risk_counts.get(True, 0))]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(labels, values)
    ax.set_ylabel("Number of students")
    for index, value in enumerate(values):
        ax.text(index, value + 4, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png", dpi=220)
    plt.close(fig)

    main = summary[
        summary["configuration"].isin(["A", "C"])
        & summary["model"].isin(
            ["Naive baseline", "Logistic Regression", "Decision Tree", "Random Forest"]
        )
    ].copy()
    model_order = ["Naive baseline", "Logistic Regression", "Decision Tree", "Random Forest"]
    config_order = ["A", "C"]

    for metric, title, filename, ylabel in [
        (
            "balanced_accuracy",
            "Balanced accuracy across feature configurations",
            "balanced_accuracy.png",
            "Balanced accuracy",
        ),
        (
            "recall_risk",
            "Recall for failing students across feature configurations",
            "failure_recall.png",
            "Recall for failing students",
        ),
    ]:
        metric_data = main[main["metric"] == metric]
        x = np.arange(len(model_order))
        width = 0.36
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        for position, config in enumerate(config_order):
            subset = metric_data[metric_data["configuration"] == config].set_index("model")
            means = [subset.loc[m, "mean"] if m in subset.index else np.nan for m in model_order]
            errors = [
                subset.loc[m, "standard_deviation"] if m in subset.index else np.nan
                for m in model_order
            ]
            ax.bar(
                x + (position - 0.5) * width,
                means,
                width,
                yerr=errors,
                capsize=3,
                label="With prior grades" if config == "A" else "Without prior grades",
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, model_order, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)

    for config, filename, title in [
        ("A", "permutation_importance_with_grades.png", "Permutation importance with prior grades"),
        ("C", "permutation_importance_without_grades.png", "Permutation importance without prior grades"),
    ]:
        subset = (
            permutation_summary[permutation_summary["configuration"] == config]
            .sort_values("mean_importance", ascending=False)
            .head(10)
            .sort_values("mean_importance")
        )
        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        ax.barh(
            subset["feature"],
            subset["mean_importance"],
            xerr=subset["standard_deviation"],
            capsize=2,
        )
        ax.set_xlabel("Mean decrease in held-out balanced accuracy")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def wide_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    wide = summary.pivot_table(
        index=["configuration", "configuration_label", "model"],
        columns="metric",
        values=["mean", "standard_deviation"],
        aggfunc="first",
    )
    wide.columns = [f"{stat}_{metric}" for stat, metric in wide.columns]
    return wide.reset_index()


def save_environment(output_dir: Path) -> None:
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "joblib": joblib.__version__,
        "random_seed": RANDOM_SEED,
    }
    (output_dir / "software_environment.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def write_manifest(output_dir: Path) -> None:
    manifest = sorted(
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "OUTPUT_MANIFEST.txt"
    )
    (output_dir / "OUTPUT_MANIFEST.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_and_validate_data(args.data)

    if args.nested_only:
        nested_summary, nested_parameters = nested_cv_analysis(df)
        nested_summary.to_csv(output_dir / "nested_cv_summary.csv", index=False)
        nested_parameters.to_csv(
            output_dir / "nested_cv_selected_hyperparameters.csv", index=False
        )
        save_environment(output_dir)
        write_manifest(output_dir)
        print(f"Nested sensitivity analysis completed. Outputs written to: {output_dir}")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    descriptive, missing = descriptive_statistics(df)
    descriptive.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    missing.to_csv(output_dir / "missing_values.csv", index=False)

    folds, summary, confusion = repeated_cv_analysis(df)
    folds.to_csv(output_dir / "repeated_cv_fold_results.csv", index=False)
    summary.to_csv(output_dir / "repeated_cv_summary_long.csv", index=False)
    wide_summary_table(summary).to_csv(
        output_dir / "repeated_cv_summary_wide.csv", index=False
    )
    confusion.to_csv(output_dir / "confusion_matrices.csv", index=False)

    comparison = original_study_comparison(summary)
    comparison.to_csv(output_dir / "original_study_comparison.csv", index=False)

    permutation_summary = permutation_analysis(df)
    permutation_summary.to_csv(
        output_dir / "permutation_importance_summary.csv", index=False
    )

    coefficients = logistic_coefficients(df)
    coefficients.to_csv(output_dir / "logistic_coefficients.csv", index=False)

    create_figures(df, summary, permutation_summary, figures_dir)
    save_environment(output_dir)
    write_manifest(output_dir)

    print(f"Analysis completed. Outputs written to: {output_dir}")
    print("Main repeated-CV summary:")
    display = wide_summary_table(summary)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
