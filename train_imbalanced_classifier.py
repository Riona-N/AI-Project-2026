"""Train and evaluate a leakage-checked classifier with training-only SMOTENC."""

from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbalancedPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "model_stage_csvs" / "final_algorithms_dataset.csv"
DEFAULT_REPORT = ROOT / "smote_classification_review.md"
TARGET = "Pollution_Level"
RANDOM_STATE = 42
CV_FOLDS = 10
TEST_FRACTION = 0.15
VALIDATION_FRACTION = 0.15

CATEGORICAL_FEATURES = ["Location"]
NUMERIC_FEATURES = [
    "pH",
    "Turbidity (NTU)",
    "Temperature (°C)",
    "DO (mg/L)",
    "BOD (mg/L)",
    "Lead (mg/L)",
    "Mercury (mg/L)",
    "Arsenic (mg/L)",
    "Hour",
    "DayOfWeek",
    "Month",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
LEAKAGE_NAME_PATTERN = re.compile(
    r"prediction|probability|(^|_)split($|_)|timestamp", re.IGNORECASE
)
SUSPICIOUS_FEATURE_PATTERN = re.compile(
    r"target|label|outcome|class|pollution", re.IGNORECASE
)


def class_counts(values: pd.Series | np.ndarray) -> dict[Any, int]:
    """Return counts in a stable class order."""
    counts = Counter(values.tolist())
    return dict(sorted(counts.items(), key=lambda item: str(item[0])))


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def make_smote_distance_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", "passthrough", CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_post_smote_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )


def make_random_forest(*, corrected: bool) -> RandomForestClassifier:
    if corrected:
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_classifier() -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor()),
            ("classifier", make_random_forest(corrected=False)),
        ]
    )


def make_sampler(y_train: pd.Series) -> SMOTENC:
    smallest_class = int(y_train.value_counts().min())
    if smallest_class < 2:
        raise ValueError(
            "SMOTENC needs at least two training examples in every class; "
            f"smallest training class has {smallest_class}."
        )
    # Keep k_neighbors valid inside CV folds when the rarest class is very small.
    k_neighbors = min(5, smallest_class - 1)
    return SMOTENC(
        categorical_features=CATEGORICAL_FEATURES,
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors,
    )


def make_smotenc_pipeline(y_train: pd.Series) -> ImbalancedPipeline:
    return ImbalancedPipeline(
        [
            ("scale_for_smote", make_smote_distance_preprocessor()),
            ("smote", make_sampler(y_train)),
            ("preprocess", make_post_smote_preprocessor()),
            ("classifier", make_random_forest(corrected=True)),
        ]
    )


def balanced_accuracy_with_labels(
    truth: pd.Series | np.ndarray,
    predictions: np.ndarray,
    labels: list[Any],
) -> float:
    matrix = confusion_matrix(truth, predictions, labels=labels)
    support = matrix.sum(axis=1)
    present_classes = support > 0
    recalls = np.diag(matrix)[present_classes] / support[present_classes]
    return float(np.mean(recalls))


def check_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    missing_columns = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Preprocessed CSV is missing required columns: {missing_columns}")

    leakage_columns = [
        column
        for column in data.columns
        if column not in FEATURES + [TARGET]
        and LEAKAGE_NAME_PATTERN.search(column)
    ]
    ignored_columns = [
        column
        for column in data.columns
        if column not in FEATURES + [TARGET] and column not in leakage_columns
    ]

    if data[TARGET].isna().any():
        raise ValueError(f"Target '{TARGET}' contains missing values.")
    if data[FEATURES].isna().any().any():
        missing = data[FEATURES].columns[data[FEATURES].isna().any()].tolist()
        raise ValueError(f"Preprocessed features contain missing values: {missing}")
    if data[TARGET].nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    duplicate_feature_rows = int(data.duplicated(subset=FEATURES, keep=False).sum())
    if duplicate_feature_rows:
        raise ValueError(
            "Identical feature rows were found "
            f"({duplicate_feature_rows} rows). Resolve duplicates before splitting "
            "so copies cannot leak across partitions."
        )

    suspicious_names = [
        feature
        for feature in FEATURES
        if SUSPICIOUS_FEATURE_PATTERN.search(feature)
    ]
    numeric_target = pd.to_numeric(data[TARGET], errors="coerce")
    high_correlations: list[str] = []
    if numeric_target.notna().all():
        for feature in NUMERIC_FEATURES:
            correlation = data[feature].corr(numeric_target)
            if pd.notna(correlation) and abs(float(correlation)) >= 0.98:
                high_correlations.append(f"{feature} (r={correlation:.3f})")

    if suspicious_names:
        warnings.warn(
            "Review feature names that resemble target-derived fields: "
            + ", ".join(suspicious_names),
            stacklevel=2,
        )
    if high_correlations:
        warnings.warn(
            "Review unusually target-correlated numeric features: "
            + ", ".join(high_correlations),
            stacklevel=2,
        )

    return data[FEATURES].copy(), data[TARGET].copy(), leakage_columns + ignored_columns


def evaluate(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    predictions = model.predict(features)
    labels = sorted(target.unique(), key=str)
    return {
        "accuracy": accuracy_score(target, predictions),
        "balanced_accuracy": balanced_accuracy_with_labels(
            target, predictions, labels
        ),
        "macro_f1": f1_score(
            target, predictions, labels=labels, average="macro", zero_division=0
        ),
        "predictions": predictions,
    }


def run_cross_validation(
    features: pd.DataFrame, target: pd.Series, labels: list[Any]
) -> dict[str, Any]:
    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    smallest_class = int(target.value_counts().min())
    cv_warning = (
        f"The rarest training class has {smallest_class} rows, fewer than "
        f"{CV_FOLDS} folds; some validation folds will not contain that class."
        if smallest_class < CV_FOLDS
        else ""
    )
    if cv_warning:
        warnings.warn(cv_warning, UserWarning, stacklevel=2)

    fold_metrics: list[dict[str, float]] = []
    for fold_number, (train_indices, valid_indices) in enumerate(
        splitter.split(features, target), start=1
    ):
        fold_train_x = features.iloc[train_indices]
        fold_train_y = target.iloc[train_indices]
        fold_valid_x = features.iloc[valid_indices]
        fold_valid_y = target.iloc[valid_indices]

        model = make_smotenc_pipeline(fold_train_y)
        model.fit(fold_train_x, fold_train_y)
        train_predictions = model.predict(fold_train_x)
        valid_predictions = model.predict(fold_valid_x)
        fold_metrics.append(
            {
                "fold": fold_number,
                "train_macro_f1": f1_score(
                    fold_train_y,
                    train_predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
                "validation_macro_f1": f1_score(
                    fold_valid_y,
                    valid_predictions,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                ),
                "validation_balanced_accuracy": balanced_accuracy_with_labels(
                    fold_valid_y, valid_predictions, labels
                ),
                "validation_support": class_counts(fold_valid_y),
            }
        )

    return {
        "folds": fold_metrics,
        "train_macro_f1_mean": float(
            np.mean([item["train_macro_f1"] for item in fold_metrics])
        ),
        "validation_macro_f1_mean": float(
            np.mean([item["validation_macro_f1"] for item in fold_metrics])
        ),
        "validation_macro_f1_std": float(
            np.std(
                [item["validation_macro_f1"] for item in fold_metrics], ddof=1
            )
        ),
        "validation_balanced_accuracy_mean": float(
            np.mean(
                [item["validation_balanced_accuracy"] for item in fold_metrics]
            )
        ),
        "warning": cv_warning,
    }


def markdown_counts(counts: dict[Any, int]) -> str:
    return ", ".join(f"`{label}`: {count:,}" for label, count in counts.items())


def markdown_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"Accuracy **{metrics['accuracy']:.3f}**; balanced accuracy "
        f"**{metrics['balanced_accuracy']:.3f}**; macro-F1 "
        f"**{metrics['macro_f1']:.3f}**"
    )


def build_report(
    *,
    source: Path,
    data: pd.DataFrame,
    features: pd.DataFrame,
    target: pd.Series,
    excluded_columns: list[str],
    train_x: pd.DataFrame,
    train_y: pd.Series,
    validation_x: pd.DataFrame,
    validation_y: pd.Series,
    test_x: pd.DataFrame,
    test_y: pd.Series,
    original_train_counts: dict[Any, int],
    resampled_counts: dict[Any, int],
    cv: dict[str, Any],
    initial_train: dict[str, Any],
    initial_validation: dict[str, Any],
    corrected_train: dict[str, Any],
    corrected_validation: dict[str, Any],
    final_test: dict[str, Any],
) -> str:
    all_labels = sorted(target.unique(), key=str)
    try:
        source_label = source.relative_to(ROOT).as_posix()
    except ValueError:
        source_label = str(source)
    confusion = confusion_matrix(test_y, final_test["predictions"], labels=all_labels)
    confusion_lines = [
        "| Actual \\ Predicted | "
        + " | ".join(f"`{label}`" for label in all_labels)
        + " |",
        "|---|"
        + "|".join("---:" for _ in all_labels)
        + "|",
    ]
    for label, row in zip(all_labels, confusion):
        confusion_lines.append(
            f"| `{label}` | " + " | ".join(str(int(value)) for value in row) + " |"
        )

    target_types = ", ".join(
        f"`{label_type.__name__}`"
        for label_type in {type(value) for value in all_labels}
    )
    excluded = ", ".join(f"`{name}`" for name in excluded_columns) or "None"
    initial_gap = initial_train["macro_f1"] - initial_validation["macro_f1"]
    corrected_gap = corrected_train["macro_f1"] - corrected_validation["macro_f1"]
    overfit_signal = initial_gap > 0.15
    underfit_signal = initial_train["macro_f1"] < 0.60
    if overfit_signal and corrected_gap <= 0.15:
        correction_result = (
            "The initial model shows an overfitting signal. The correction reduced "
            f"the train/validation macro-F1 gap from {initial_gap:.3f} to "
            f"{corrected_gap:.3f}, below the 0.15 threshold."
        )
    elif overfit_signal:
        correction_result = (
            "The initial model shows an overfitting signal; the corrected model "
            f"still has a macro-F1 gap of {corrected_gap:.3f}."
        )
    else:
        correction_result = (
            "The initial model does not exceed the configured overfit-gap threshold "
            f"(0.15); corrected model macro-F1 gap is {corrected_gap:.3f}."
        )
    if underfit_signal:
        correction_result += (
            " The initial model also falls below the configured training macro-F1 "
            "underfitting threshold (0.60); this needs review alongside the final "
            "test metrics."
        )

    fold_lines = [
        "| Fold | Train macro-F1 | Validation macro-F1 | Validation balanced accuracy | Validation class counts |",
        "|---:|---:|---:|---:|---|",
    ]
    for fold in cv["folds"]:
        fold_support = markdown_counts(fold["validation_support"])
        fold_lines.append(
            f"| {fold['fold']} | {fold['train_macro_f1']:.3f} | "
            f"{fold['validation_macro_f1']:.3f} | "
            f"{fold['validation_balanced_accuracy']:.3f} | {fold_support} |"
        )

    class_report = classification_report(
        test_y,
        final_test["predictions"],
        labels=all_labels,
        zero_division=0,
        digits=3,
    )
    return f"""# SMOTE Classification Review

## Scope and input

- **Input:** `{source_label}` (the branch's preprocessed final model-stage CSV)
- **Original observations:** {len(data):,}
- **Target:** `{TARGET}`; stored dtype `{target.dtype}`; observed labels: {", ".join(f"`{label}`" for label in all_labels)} (type: {target_types})
- **Predictive features:** {features.shape[1]} ({len(CATEGORICAL_FEATURES)} categorical, {len(NUMERIC_FEATURES)} numeric)
- **Feature names:** {", ".join(f"`{name}`" for name in FEATURES)}
- **Original target counts:** {markdown_counts(class_counts(target))}
- **Excluded auxiliary columns:** {excluded}

The existing `Dataset_Split` and FT-Transformer prediction/probability columns are not model inputs. This workflow starts from the preprocessed features, ignores the former holdout assignment, and makes a new stratified split. It does not use raw `Timestamp` or repeat preprocessing from the raw CSV.

## Leakage and data checks

- `Pollution_Level` is selected only as the target and is never in `X`.
- Metadata and prediction/probability fields are excluded; the feature set is an explicit allow-list from the preprocessing stage.
- Exact duplicate feature rows across the dataset: **0** (the script stops if any are found to prevent duplicate leakage).
- Missing values in the selected target/features: **0** (the script stops if any are found).
- Suspicious feature-name checks and numeric feature/target correlations are run at startup; any warning must be reviewed.
- The full dataset is split before fitting a sampler. Numeric inputs are scaled using the training fold only before SMOTENC's neighbor search, while `Location` remains categorical. The held-out validation and test partitions are never resampled.

## Split and SMOTE sizes

Stratified split, random state {RANDOM_STATE}: **70% train / 15% validation / 15% test** (rounded to whole observations).

| Partition / stage | Rows before SMOTE | Rows after SMOTE | Class counts |
|---|---:|---:|---|
| Training set | {len(train_x):,} | — | {markdown_counts(original_train_counts)} |
| Training set after SMOTENC | {len(train_x):,} | {sum(resampled_counts.values()):,} | {markdown_counts(resampled_counts)} |
| Validation set (untouched) | {len(validation_x):,} | — | {markdown_counts(class_counts(validation_y))} |
| Final test set (untouched) | {len(test_x):,} | — | {markdown_counts(class_counts(test_y))} |

SMOTENC preserves the categorical `Location` field and synthesizes numeric inputs. Its `k_neighbors` is capped at 5 and reduced when needed to remain below the smallest class count in each training fold. It is applied only to training data. The rarest training class has very few original examples, so its synthetic samples do not replace the need for more real labeled data.

## 10-fold cross-validation

SMOTENC is fitted separately inside each training fold; validation-fold observations are never resampled. Macro-F1 is the primary comparison metric because accuracy is misleading for the class imbalance.

| Cross-validation summary | Result |
|---|---:|
| Mean training macro-F1 | {cv["train_macro_f1_mean"]:.3f} |
| Mean validation macro-F1 | {cv["validation_macro_f1_mean"]:.3f} ± {cv["validation_macro_f1_std"]:.3f} |
| Mean validation balanced accuracy | {cv["validation_balanced_accuracy_mean"]:.3f} |

{cv["warning"] or "Every class is represented sufficiently for the requested fold count."}

{chr(10).join(fold_lines)}

## Overfitting check and correction

The initial model is an unresampled, unrestricted Random Forest. Its macro-F1 is measured on the original (not oversampled) training data and validation data. An overfit signal is defined here as a training-minus-validation macro-F1 gap greater than **0.15**; an underfit signal is training macro-F1 below **0.60**.

| Model | Training macro-F1 | Validation macro-F1 | Gap |
|---|---:|---:|---:|
| Initial Random Forest | {initial_train["macro_f1"]:.3f} | {initial_validation["macro_f1"]:.3f} | {initial_gap:.3f} |
| Corrected SMOTENC + regularized Random Forest | {corrected_train["macro_f1"]:.3f} | {corrected_validation["macro_f1"]:.3f} | {corrected_gap:.3f} |

{correction_result}

Correction applied: SMOTENC on training data plus a strongly depth-limited Random Forest (`max_depth=3`, `min_samples_leaf=5`, `min_samples_split=10`). The final model is refit on the training partition only; validation is used for diagnosis and the test set remains untouched until final evaluation.

## Final test evaluation

{markdown_metrics(final_test)}.

### Per-class test metrics

```text
{class_report.rstrip()}
```

### Confusion matrix (counts)

{chr(10).join(confusion_lines)}

## Reproducibility and limitations

- Run from the repository root with: `python train_imbalanced_classifier.py`
- Optional input path: `python train_imbalanced_classifier.py path\\to\\preprocessed.csv`
- Random seed: {RANDOM_STATE}; CV folds: {CV_FOLDS}; results generated with the current installed pandas/scikit-learn/imbalanced-learn versions.
- There are only **{class_counts(target).get(all_labels[0], 0)}** original observations in the rarest class. A 10-fold split cannot place this class in every validation fold, and scores for that class are consequently unstable. Report the class counts and per-class metrics with all conclusions.
- This is a random stratified split of hourly data. For a claim about future-time performance, repeat the evaluation with a chronological holdout and account for time-dependent leakage.
"""


def main() -> None:
    import sys

    source = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_INPUT
    if not source.is_absolute():
        source = (Path.cwd() / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Preprocessed CSV not found: {source}")

    data = pd.read_csv(source)
    features, target, excluded_columns = check_dataset(data)
    labels = sorted(target.unique(), key=str)

    train_x, holdout_x, train_y, holdout_y = train_test_split(
        features,
        target,
        test_size=TEST_FRACTION + VALIDATION_FRACTION,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    validation_x, test_x, validation_y, test_y = train_test_split(
        holdout_x,
        holdout_y,
        test_size=TEST_FRACTION / (TEST_FRACTION + VALIDATION_FRACTION),
        stratify=holdout_y,
        random_state=RANDOM_STATE,
    )

    print(f"Input: {source}")
    print(f"Dataset: {len(data):,} rows; {features.shape[1]} features")
    print(f"Target classes and original counts: {markdown_counts(class_counts(target))}")
    print(f"Train / validation / test rows: {len(train_x)} / {len(validation_x)} / {len(test_x)}")
    print(f"Before SMOTE (training only): {len(train_x)} rows; {markdown_counts(class_counts(train_y))}")

    cv = run_cross_validation(train_x, train_y, labels)

    initial_model = make_classifier()
    initial_model.fit(train_x, train_y)
    initial_train = evaluate(initial_model, train_x, train_y)
    initial_validation = evaluate(initial_model, validation_x, validation_y)

    distance_preprocessor = make_smote_distance_preprocessor()
    scaled_train_x = distance_preprocessor.fit_transform(train_x, train_y)
    sampler = make_sampler(train_y)
    resampled_x, resampled_y = sampler.fit_resample(scaled_train_x, train_y)
    original_train_counts = class_counts(train_y)
    resampled_counts = class_counts(resampled_y)
    print(
        f"After SMOTE (training only): {len(resampled_x)} rows; "
        f"{markdown_counts(resampled_counts)}"
    )

    corrected_model = make_smotenc_pipeline(train_y)
    corrected_model.fit(train_x, train_y)
    corrected_train = evaluate(corrected_model, train_x, train_y)
    corrected_validation = evaluate(corrected_model, validation_x, validation_y)
    final_test = evaluate(corrected_model, test_x, test_y)

    report = build_report(
        source=source,
        data=data,
        features=features,
        target=target,
        excluded_columns=excluded_columns,
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=test_y,
        original_train_counts=original_train_counts,
        resampled_counts=resampled_counts,
        cv=cv,
        initial_train=initial_train,
        initial_validation=initial_validation,
        corrected_train=corrected_train,
        corrected_validation=corrected_validation,
        final_test=final_test,
    )
    DEFAULT_REPORT.write_text(report, encoding="utf-8")

    print(f"10-fold CV mean validation macro-F1: {cv['validation_macro_f1_mean']:.3f}")
    print(f"Initial validation: {markdown_metrics(initial_validation)}")
    print(f"Corrected validation: {markdown_metrics(corrected_validation)}")
    print(f"Final test: {markdown_metrics(final_test)}")
    


if __name__ == "__main__":
    main()
