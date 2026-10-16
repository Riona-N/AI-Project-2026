# SMOTE Classification Review

## Scope and input

- **Input:** `model_stage_csvs/final_algorithms_dataset.csv` (the branch's preprocessed final model-stage CSV)
- **Original observations:** 1,000
- **Target:** `Pollution_Level`; stored dtype `int64`; observed labels: `0`, `1`, `2` (type: `int64`)
- **Predictive features:** 12 (1 categorical, 11 numeric)
- **Feature names:** `Location`, `pH`, `Turbidity (NTU)`, `Temperature (°C)`, `DO (mg/L)`, `BOD (mg/L)`, `Lead (mg/L)`, `Mercury (mg/L)`, `Arsenic (mg/L)`, `Hour`, `DayOfWeek`, `Month`
- **Original target counts:** `0`: 6, `1`: 79, `2`: 915
- **Excluded auxiliary columns:** `Dataset_Split`, `ft_transformer_prediction`, `ft_transformer_probability_class_0`, `ft_transformer_probability_class_1`, `ft_transformer_probability_class_2`

The existing `Dataset_Split` and FT-Transformer prediction/probability columns are not model inputs. This workflow starts from the preprocessed features, ignores the former holdout assignment, and makes a new stratified split. It does not use raw `Timestamp` or repeat preprocessing from the raw CSV.

## Leakage and data checks

- `Pollution_Level` is selected only as the target and is never in `X`.
- Metadata and prediction/probability fields are excluded; the feature set is an explicit allow-list from the preprocessing stage.
- Exact duplicate feature rows across the dataset: **0** (the script stops if any are found to prevent duplicate leakage).
- Missing values in the selected target/features: **0** (the script stops if any are found).
- Suspicious feature-name checks and numeric feature/target correlations are run at startup; any warning must be reviewed.
- The full dataset is split before fitting a sampler. Numeric inputs are scaled using the training fold only before SMOTENC's neighbor search, while `Location` remains categorical. The held-out validation and test partitions are never resampled.

## Split and SMOTE sizes

Stratified split, random state 42: **70% train / 15% validation / 15% test** (rounded to whole observations).

| Partition / stage | Rows before SMOTE | Rows after SMOTE | Class counts |
|---|---:|---:|---|
| Training set | 700 | — | `0`: 4, `1`: 55, `2`: 641 |
| Training set after SMOTENC | 700 | 1,923 | `0`: 641, `1`: 641, `2`: 641 |
| Validation set (untouched) | 150 | — | `0`: 1, `1`: 12, `2`: 137 |
| Final test set (untouched) | 150 | — | `0`: 1, `1`: 12, `2`: 137 |

SMOTENC preserves the categorical `Location` field and synthesizes numeric inputs. Its `k_neighbors` is capped at 5 and reduced when needed to remain below the smallest class count in each training fold. It is applied only to training data. The rarest training class has very few original examples, so its synthetic samples do not replace the need for more real labeled data.

## 10-fold cross-validation

SMOTENC is fitted separately inside each training fold; validation-fold observations are never resampled. Macro-F1 is the primary comparison metric because accuracy is misleading for the class imbalance.

| Cross-validation summary | Result |
|---|---:|
| Mean training macro-F1 | 0.728 |
| Mean validation macro-F1 | 0.518 ± 0.082 |
| Mean validation balanced accuracy | 0.781 |

The rarest training class has 4 rows, fewer than 10 folds; some validation folds will not contain that class.

| Fold | Train macro-F1 | Validation macro-F1 | Validation balanced accuracy | Validation class counts |
|---:|---:|---:|---:|---|
| 1 | 0.706 | 0.539 | 0.954 | `1`: 5, `2`: 65 |
| 2 | 0.738 | 0.490 | 0.786 | `1`: 6, `2`: 64 |
| 3 | 0.706 | 0.510 | 0.802 | `1`: 6, `2`: 64 |
| 4 | 0.722 | 0.611 | 0.977 | `1`: 6, `2`: 64 |
| 5 | 0.712 | 0.570 | 0.961 | `1`: 6, `2`: 64 |
| 6 | 0.702 | 0.525 | 0.945 | `1`: 6, `2`: 64 |
| 7 | 0.737 | 0.471 | 0.564 | `0`: 1, `1`: 5, `2`: 64 |
| 8 | 0.804 | 0.358 | 0.369 | `0`: 1, `1`: 5, `2`: 64 |
| 9 | 0.746 | 0.644 | 0.892 | `0`: 1, `1`: 5, `2`: 64 |
| 10 | 0.712 | 0.457 | 0.558 | `0`: 1, `1`: 5, `2`: 64 |

## Overfitting check and correction

The initial model is an unresampled, unrestricted Random Forest. Its macro-F1 is measured on the original (not oversampled) training data and validation data. An overfit signal is defined here as a training-minus-validation macro-F1 gap greater than **0.15**; an underfit signal is training macro-F1 below **0.60**.

| Model | Training macro-F1 | Validation macro-F1 | Gap |
|---|---:|---:|---:|
| Initial Random Forest | 1.000 | 0.538 | 0.462 |
| Corrected SMOTENC + regularized Random Forest | 0.721 | 0.638 | 0.083 |

The initial model shows an overfitting signal. The correction reduced the train/validation macro-F1 gap from 0.462 to 0.083, below the 0.15 threshold.

Correction applied: SMOTENC on training data plus a strongly depth-limited Random Forest (`max_depth=3`, `min_samples_leaf=5`, `min_samples_split=10`). The final model is refit on the training partition only; validation is used for diagnosis and the test set remains untouched until final evaluation.

## Final test evaluation

Accuracy **0.920**; balanced accuracy **0.945**; macro-F1 **0.763**.

### Per-class test metrics

```text
              precision    recall  f1-score   support

           0      0.500     1.000     0.667         1
           1      0.524     0.917     0.667        12
           2      0.992     0.920     0.955       137

    accuracy                          0.920       150
   macro avg      0.672     0.945     0.763       150
weighted avg      0.951     0.920     0.930       150
```

### Confusion matrix (counts)

| Actual \ Predicted | `0` | `1` | `2` |
|---|---:|---:|---:|
| `0` | 1 | 0 | 0 |
| `1` | 0 | 11 | 1 |
| `2` | 1 | 10 | 126 |

## Reproducibility and limitations

- Run from the repository root with: `python train_imbalanced_classifier.py`
- Optional input path: `python train_imbalanced_classifier.py path\to\preprocessed.csv`
- Random seed: 42; CV folds: 10; results generated with the current installed pandas/scikit-learn/imbalanced-learn versions.
- There are only **6** original observations in the rarest class. A 10-fold split cannot place this class in every validation fold, and scores for that class are consequently unstable. Report the class counts and per-class metrics with all conclusions.
- This is a random stratified split of hourly data. For a claim about future-time performance, repeat the evaluation with a chronological holdout and account for time-dependent leakage.
