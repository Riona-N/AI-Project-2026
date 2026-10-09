# Water Quality Classification: Model Comparison Report

**Purpose:** Summarize the dataset, preprocessing, and previously run model comparisons for review.  
**Dataset:** `Water_Quality_Dataset.csv`  
**Target:** `Pollution_Level` (three classes: 0, 1, and 2)

## Executive summary

- The source CSV contains **1,000 observations and 11 columns**. No missing values or exact duplicate rows were found.
- Preprocessing removed the raw `Timestamp` feature and derived `Hour`, `DayOfWeek`, and `Month` from it. No observations were removed. The resulting model input has **12 features** plus the target, for **13 columns total**.
- All model comparisons used those same 12 input features and the same stratified 80/20 split (random state 42). The split has 800 training rows and 200 held-out rows.
- The best recorded accuracy was **92.00%** for FT-Transformer and MLP in the two-model rerun; FT-Transformer also scored **92.00%** in the final rerun.
- The target is severely imbalanced: class 2 accounts for 91.5% of observations. Predicting class 2 for every held-out row would achieve **91.50% accuracy**. Thus 92.00% is only 0.50 percentage points above that baseline, and accuracy alone is not enough to establish useful minority-class performance.
- The reported holdout rows were also passed to training as the validation set for checkpoint selection. These results should therefore be treated as **validation/holdout accuracy, not an independent final test estimate**.

## 1. Dataset before preprocessing

### Dataset dimensions and quality checks

| Measure | Before preprocessing |
|---|---:|
| Rows / observations | 1,000 |
| Columns | 11 |
| Missing values | 0 |
| Exact duplicate rows | 0 |
| Parsed timestamp failures | 0 |
| Duplicate timestamps | 0 |
| Timestamp range | 2024-01-01 00:00 through 2024-02-11 15:00 |
| Sampling interval | Hourly |

### Original columns

| Column | Role / type |
|---|---|
| `Timestamp` | Date/time string; later used to derive time features |
| `Location` | Categorical input feature |
| `pH` | Numeric sensor feature |
| `Turbidity (NTU)` | Numeric sensor feature |
| `Temperature (°C)` | Numeric sensor feature |
| `DO (mg/L)` | Numeric sensor feature |
| `BOD (mg/L)` | Numeric sensor feature |
| `Lead (mg/L)` | Numeric sensor feature |
| `Mercury (mg/L)` | Numeric sensor feature |
| `Arsenic (mg/L)` | Numeric sensor feature |
| `Pollution_Level` | **Target**; class label 0, 1, or 2 |

### Target class balance

| `Pollution_Level` | Rows | Share |
|---:|---:|---:|
| 0 | 6 | 0.6% |
| 1 | 79 | 7.9% |
| 2 | 915 | 91.5% |
| **Total** | **1,000** | **100.0%** |

The majority-class baseline is therefore **91.5% accuracy** when predicting class 2 for every row.

### Original numeric-column descriptive statistics

These are statistics before model normalization. `Pollution_Level` is included for completeness but is the target, not an input feature.

| Numeric column | Mean | Std. dev. | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| `pH` | 7.2509 | 1.0245 | 5.5162 | 7.2652 | 8.9979 |
| `Turbidity (NTU)` | 10.2187 | 5.6316 | 0.5026 | 10.2369 | 19.9678 |
| `Temperature (°C)` | 24.9669 | 5.7566 | 15.0002 | 25.0415 | 34.9912 |
| `DO (mg/L)` | 5.9286 | 2.2875 | 2.0002 | 5.8713 | 9.9820 |
| `BOD (mg/L)` | 5.4830 | 2.6041 | 1.0085 | 5.3689 | 9.9942 |
| `Lead (mg/L)` | 0.009965 | 0.005746 | 0.000128 | 0.009894 | 0.019989 |
| `Mercury (mg/L)` | 0.000981 | 0.000569 | 0.000010 | 0.000977 | 0.001998 |
| `Arsenic (mg/L)` | 0.009855 | 0.005544 | 0.000505 | 0.009392 | 0.019922 |
| `Pollution_Level` (target) | 1.9090 | 0.3079 | 0 | 2 | 2 |

## 2. Preprocessing and model features

### Changes to columns

| Operation | Result |
|---|---|
| Removed `Timestamp` | The original timestamp string was not passed as a raw feature to the models. |
| Derived `Hour` | Hour of day, 0–23. |
| Derived `DayOfWeek` | Day of week, 0–6. |
| Derived `Month` | Month number, 1–12. |
| Kept `Location` | Passed as the categorical feature. |
| Kept eight sensor columns | Passed as continuous features. |
| Kept `Pollution_Level` | Remained the target; not an input feature. |
| Removed rows | None. |
| Removed missing-value rows | None were missing. |
| Removed redundant columns | None. The correlation check found no numeric-feature pair with absolute correlation greater than 0.90. |

The one input column `Timestamp` was replaced by three derived columns. Consequently, preprocessing increased the total column count from 11 to 13; it did not perform dimensionality reduction.

### Model-ready columns and features

**12 input features:**

- Categorical: `Location`
- Continuous: `pH`, `Turbidity (NTU)`, `Temperature (°C)`, `DO (mg/L)`, `BOD (mg/L)`, `Lead (mg/L)`, `Mercury (mg/L)`, `Arsenic (mg/L)`, `Hour`, `DayOfWeek`, `Month`

**Target:** `Pollution_Level`

**Model-ready dataset:** 1,000 rows × 13 columns (12 features + 1 target).

Continuous features were configured for quantile-normal transformation and normalization in PyTorch Tabular. `Location` was provided as categorical. The training transformation is fit by the modeling pipeline; the dataset statistics above describe the raw measurements, not the transformed values.

### Train/validation split

An 80/20 stratified split with random state 42 was used. The 200-row held-out partition was supplied as `validation` during `fit` and was also used to calculate the reported accuracy.

| Partition | Rows | Class 0 | Class 1 | Class 2 |
|---|---:|---:|---:|---:|
| Training | 800 | 5 | 63 | 732 |
| Validation / reported holdout | 200 | 1 | 16 | 183 |

### Dataset size and columns at each model-comparison stage

The algorithm comparisons are successive comparisons, not sequential transformations of the CSV. The dataset was not changed between comparison rounds.

| Stage | Full pre-split data | Training partition | Validation / reported holdout | Input features | Target |
|---|---:|---:|---:|---:|---|
| After running 5 algorithms | 1,000 × 13 | 800 × 13 | 200 × 13 | Same 12 features listed above | `Pollution_Level` |
| After running 3 algorithms | 1,000 × 13 | 800 × 13 | 200 × 13 | Same 12 features listed above | `Pollution_Level` |
| After running 2 algorithms | 1,000 × 13 | 800 × 13 | 200 × 13 | Same 12 features listed above | `Pollution_Level` |
| Final FT-Transformer run | 1,000 × 13 | 800 × 13 | 200 × 13 | Same 12 features listed above | `Pollution_Level` |

In each shape, rows × columns are shown. All 13 columns are the 12 inputs plus the target; the target is excluded from the model input.

## 3. Five-algorithm comparison

Each algorithm used the same preprocessed feature columns and split. No algorithm removed or added dataset columns.

| Rank by recorded accuracy | Algorithm | Accuracy | Correct of 200 |
|---:|---|---:|---:|
| 1 (tie) | FT-Transformer | 92.00% | 184 |
| 1 (tie) | MLP (Category Embedding) | 92.00% | 184 |
| 3 (tie) | TabTransformer | 91.50% | 183 |
| 3 (tie) | Deep & Cross Network (DANet implementation) | 91.50% | 183 |
| 5 | TabNet | 80.00% | 160 |

The top three slots include ties: FT-Transformer and MLP tie for first; TabTransformer and Deep & Cross Network tie for the next score.

## 4. Three-algorithm rerun

The rerun compared FT-Transformer, MLP, and TabTransformer on the same 12 features and the same 200-row validation/holdout partition.

| Rank by recorded accuracy | Algorithm | Accuracy | Correct of 200 |
|---:|---|---:|---:|
| 1 (tie) | FT-Transformer | 92.00% | 184 |
| 1 (tie) | MLP (Category Embedding) | 92.00% | 184 |
| 3 | TabTransformer | 91.50% | 183 |

## 5. Two-algorithm rerun

The rerun compared FT-Transformer and MLP on the same preprocessed data and split.

| Rank by recorded accuracy | Algorithm | Accuracy | Correct of 200 |
|---:|---|---:|---:|
| 1 (tie) | FT-Transformer | 92.00% | 184 |
| 1 (tie) | MLP (Category Embedding) | 92.00% | 184 |

Neither model was higher in this recorded run.

## 6. Final FT-Transformer rerun

The final run used FT-Transformer with the same preprocessing, 12 input features, and 80/20 stratified split.

| Model | Accuracy | Correct of 200 |
|---|---:|---:|
| FT-Transformer | **92.00%** | **184** |

Final recorded accuracy: **92.00%**.

## 7. Feature variance and importance check

A separate quick check was run after timestamp feature engineering. VarianceThreshold retained all 12 input features; it found no zero-variance feature. A Random Forest fit on the training partition was used to get an indicative feature-importance ranking.

| Feature | Random Forest importance |
|---|---:|
| `Turbidity (NTU)` | 0.2190 |
| `BOD (mg/L)` | 0.1835 |
| `Mercury (mg/L)` | 0.1118 |
| `Arsenic (mg/L)` | 0.1029 |
| `Lead (mg/L)` | 0.1026 |
| `pH` | 0.0768 |
| `DO (mg/L)` | 0.0703 |
| `Temperature (°C)` | 0.0552 |
| `Hour` | 0.0327 |
| `Location` | 0.0197 |
| `DayOfWeek` | 0.0196 |
| `Month` | 0.0059 |

`Month` had the lowest importance in this quick check, but no feature was removed. These are Random Forest importances, not FT-Transformer importances, and are not proof that a feature should be dropped. No feature-importance experiment comparing the FT-Transformer with and without `Month` was recorded.

## 8. Interpretation and limitations

1. **Severe class imbalance:** 91.5% of rows are class 2. A 91.5% majority-class baseline is already close to the reported 92.0% accuracy. Per-class precision, recall, F1, balanced accuracy, and a confusion matrix are needed to determine whether classes 0 and 1 are being identified.
2. **Validation partition reuse:** The holdout partition was passed as `validation` during fitting (the trainer used validation-based checkpointing) and then scored. It is not an untouched test set, so the recorded accuracy may be optimistic.
3. **Temporal structure:** Rows are hourly and the split was random, not chronological. Nearby observations can occur in both partitions. For estimating future-time performance, use a chronological split (train on earlier timestamps, validate/test on later timestamps).
4. **Small minority-class sample:** The validation partition contains only one class-0 observation, so its class-0 performance cannot be estimated reliably.
5. **Run scope:** The recorded model comparison used three training epochs. Accuracy is the only model evaluation metric consistently captured in the previous runs; no historical confusion matrices or precision/recall/F1 values were retained.
6. **Repeated comparisons:** Small differences such as 92.0% versus 91.5% represent one correct validation prediction on a 200-row partition. They should not be treated as evidence of a robust performance advantage without repeated seeds or cross-validation.

## Conclusion

The final recorded FT-Transformer accuracy is **92.00%** on the reused 200-row validation/holdout partition. FT-Transformer and MLP tied at 92.00% in the recorded two-model rerun. No input columns were removed by preprocessing or by the variance check; `Timestamp` was replaced by its three derived time features.

For a stronger professor-review result, the next evaluation should use a chronological untouched test set and report accuracy together with balanced accuracy, macro-F1, per-class precision/recall, and a confusion matrix. Class imbalance should also be addressed or explicitly considered.
