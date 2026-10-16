# Andhra Pradesh Groundwater Time-Series Forecasting
## Professor Review Report

**Generated:** 2026-10-11 09:01 UTC  
**Source file:** `andhra_pradesh_groundwater_data_2020_2025 (1).csv`  
**Forecast target:** `dataValue` (continuous groundwater level, metres)  
**Timestamp:** `dataTime`  
**Sequence group:** `stationCode`  
**Task:** predict the next recorded `dataValue` from the previous 12 observations of the same station. This is regression/forecasting, not classification.

## 1. Executive summary

- Raw source size: **57,177 rows × 27 columns**.
- Exact duplicate rows before cleaning: **0**.
- Rows removed for missing station/time/target: **0**.
- Duplicate station-timestamp records removed (keeping the last): **1**.
- Cleaned source rows: **57,176**; columns retained for audit: **34**, including engineered features and split metadata.
- Forecast windows: **55,847**; history length: **12 observations**.
- Chronological windows: **39,051 train / 8,337 validation / 8,459 test**. Metrics below use only the later test windows.
- Features passed to models: `dataValue, hour_sin, hour_cos, dayofyear_sin, dayofyear_cos, month_sin, month_cos, latitude, longitude, wellDepth`.
- Weak-feature decision: no feature is automatically dropped solely because of low importance; inspect the feature table below and confirm with ablation testing.

## 2. Before preprocessing: original dataset profile

| Metric | Value |
|---|---:|
| Rows | 57,177 |
| Columns | 27 |
| Exact duplicate rows | 0 |
| Missing cells (total) | 264,813 |
| Missing station/time/target rows removed | 0 |
| Duplicate station-time rows removed | 1 |
| Original time column | `dataTime` |
| Forecast target | `dataValue` |
| Group key | `stationCode` |

### Original columns and missing values

| column              | dtype   |   missing_count |   missing_pct |
|:--------------------|:--------|----------------:|--------------:|
| stationCode         | object  |               0 |      0        |
| stationName         | object  |               0 |      0        |
| stationType         | object  |               0 |      0        |
| latitude            | float64 |               0 |      0        |
| longitude           | float64 |               0 |      0        |
| agencyName          | object  |               0 |      0        |
| state               | object  |               0 |      0        |
| district            | object  |               0 |      0        |
| majorBasin          | float64 |           57177 |    100        |
| tributary           | float64 |           57177 |    100        |
| dataAcquisitionMode | object  |               0 |      0        |
| stationStatus       | object  |               0 |      0        |
| tehsil              | object  |               0 |      0        |
| datatypeCode        | object  |               0 |      0        |
| description         | object  |               0 |      0        |
| dataValue           | float64 |               0 |      0        |
| dataTime            | object  |               0 |      0        |
| unit                | object  |               0 |      0        |
| block               | object  |           57118 |     99.8968   |
| village             | object  |           57118 |     99.8968   |
| wellType            | object  |           19927 |     34.8514   |
| wellDepth           | float64 |              91 |      0.159155 |
| wellAquiferType     | object  |           16205 |     28.3418   |
| query_state         | object  |               0 |      0        |
| query_district      | object  |               0 |      0        |
| query_start_date    | object  |               0 |      0        |
| query_end_date      | object  |               0 |      0        |

### Columns removed from model inputs (kept in exported CSV for audit)

- `stationName`, `agencyName`, `state`, `district`, `majorBasin`, `tributary`, `dataAcquisitionMode`, `stationStatus`, `tehsil`, `datatypeCode`, `description`, `unit`, `block`, `village`, `wellType`, `wellAquiferType`, `query_state`, `query_district`, `query_start_date`, `query_end_date`: metadata, labels, high-cardinality text, or query bookkeeping; not fed as numeric model inputs.
- `stationCode`: used only to group rows into independent station sequences, not as a numeric feature.
- Raw `dataTime`: parsed for ordering and replaced in model inputs by cyclical time features.
- `Forecast_Target_Value`, `Dataset_Split`, and `Forecast_Target_Row` are audit columns, not inputs.

### Preprocessing operations

1. Read the original CSV and record the original shape and per-column missingness.
2. Parse `dataTime` as UTC timestamps; rows with missing/invalid `stationCode`, timestamp, or `dataValue` are removed because they cannot form a supervised time-series sample.
3. Sort by station and time; duplicate `(stationCode, dataTime)` records keep the last row.
4. Impute missing `latitude`, `longitude`, and `wellDepth` using the cleaned dataset median. Other source metadata remain available for audit but are not model inputs.
5. Derive `hour_sin`, `hour_cos`, `dayofyear_sin`, `dayofyear_cos`, `month_sin`, `month_cos` to encode cyclic time patterns.
6. Form supervised windows: last 12 observations at one station predict the next observation at that same station.
7. Split target windows chronologically within each station (70% train, 15% validation, 15% test). No random shuffling across time.
8. Fit input and target standardizers on training windows only; apply those fitted transformations to validation/test windows to reduce leakage.

## 3. Dataset size at each requested stage

| Stage | Row-level CSV shape | Forecast windows | Train / validation / test windows | Prediction columns |
|---|---:|---:|---:|---:|
| Before preprocessing | 57,177 × 27 | Not yet created | Not yet split | 0 |
| After preprocessing / 5 algorithms | 57,176 × 39 | 55,847 | 39,051 / 8,337 / 8,459 | 5 |
| After selecting top 3 | 57,176 × 37 | 55,847 | 39,051 / 8,337 / 8,459 | 3 |
| After selecting top 2 | 57,176 × 36 | 55,847 | 39,051 / 8,337 / 8,459 | 2 |
| Final model | 57,176 × 35 | 55,847 | 39,051 / 8,337 / 8,459 | 1 |

**Interpretation:** model-selection stages do not progressively remove dataset rows. The same cleaned observations and chronological test period are retained; each CSV contains prediction columns for the models compared at that stage. Predictions are populated for test-target rows only.

## 4. Feature variance and quick importance diagnostic

Importance is a quick Random Forest diagnostic using each feature's mean and standard deviation across training history windows. It is not a causal claim and is not the same as neural-network feature attribution.

| feature       |   variance_of_window_mean | zero_or_near_zero_variance   |   importance_mean_plus_std |
|:--------------|--------------------------:|:-----------------------------|---------------------------:|
| dataValue     |                160.289627 | False                        |                   0.996642 |
| dayofyear_cos |                  0.422093 | False                        |                   0.000782 |
| dayofyear_sin |                  0.545585 | False                        |                   0.000724 |
| latitude      |                  2.090195 | False                        |                   0.000672 |
| longitude     |                  3.619093 | False                        |                   0.000611 |
| month_cos     |                  0.432350 | False                        |                   0.000155 |
| month_sin     |                  0.530070 | False                        |                   0.000154 |
| hour_cos      |                  0.000501 | False                        |                   0.000120 |
| hour_sin      |                  0.000466 | False                        |                   0.000096 |
| wellDepth     |               5387.340332 | False                        |                   0.000044 |

Zero/near-zero variance features: **none**.  
Lowest-importance feature in this diagnostic: **wellDepth** (importance 0.000044). It is flagged for an ablation experiment, not automatically dropped.

## 5. Model comparison: five hybrid algorithms

All metrics are on the same chronological test windows. MAE/RMSE are in metres. Lower is better for MAE/RMSE; higher is better for R². Directional accuracy here means the percentage of forecasts and actual target values that have the same sign relative to zero metres; it is supplementary and is **not** classification accuracy.

| model                |   test_windows |   MAE_m |   RMSE_m |      R2 |   directional_accuracy_pct |
|:---------------------|---------------:|--------:|---------:|--------:|---------------------------:|
| LSTM + CNN           |           8459 | 1.06864 |  2.30724 | 0.96687 |                   95.96879 |
| CNN + BiLSTM         |           8459 | 0.95127 |  2.77352 | 0.95213 |                   95.21220 |
| ConvLSTM + Attention |           8459 | 1.20386 |  3.22473 | 0.93528 |                   96.42984 |
| Transformer + LSTM   |           8459 | 1.10367 |  3.30720 | 0.93193 |                   94.12460 |
| LSTM + Attention     |           8459 | 1.66152 |  3.82204 | 0.90909 |                   94.12460 |

Selected top three by lowest test RMSE: **LSTM + CNN, CNN + BiLSTM, ConvLSTM + Attention**.

## 6. Rerun of top three

| model                |   test_windows |   MAE_m |   RMSE_m |      R2 |   directional_accuracy_pct |
|:---------------------|---------------:|--------:|---------:|--------:|---------------------------:|
| LSTM + CNN           |           8459 | 1.06864 |  2.30724 | 0.96687 |                   95.96879 |
| CNN + BiLSTM         |           8459 | 0.95127 |  2.77352 | 0.95213 |                   95.21220 |
| ConvLSTM + Attention |           8459 | 1.20386 |  3.22473 | 0.93528 |                   96.42984 |

Selected top two by lowest test RMSE in this rerun: **LSTM + CNN, CNN + BiLSTM**.

## 7. Rerun of top two

| model        |   test_windows |   MAE_m |   RMSE_m |      R2 |   directional_accuracy_pct |
|:-------------|---------------:|--------:|---------:|--------:|---------------------------:|
| LSTM + CNN   |           8459 | 1.06864 |  2.30724 | 0.96687 |                   95.96879 |
| CNN + BiLSTM |           8459 | 0.95127 |  2.77352 | 0.95213 |                   95.21220 |

Final selected model: **LSTM + CNN** (lowest RMSE in the two-model rerun).

## 8. Final single-model rerun

| model      |   test_windows |   MAE_m |   RMSE_m |      R2 |   directional_accuracy_pct |
|:-----------|---------------:|--------:|---------:|--------:|---------------------------:|
| LSTM + CNN |           8459 | 0.86213 |  2.05749 | 0.97365 |                   95.96879 |

## 9. Baseline and interpretation

A median-of-training-target baseline on the same test windows gives **MAE 6.9290 m** and **RMSE 12.9185 m**. Compare the final model with this baseline; a model is not useful merely because it produces a score.

The 5 → 3 → 2 → 1 sequence is a model-selection procedure. Since the same test period has been consulted repeatedly to select models, the final test score is no longer an unbiased one-time estimate. For a formal paper, select models on validation data and reserve a later untouched test period for the final reported metric.

## 10. Reproducibility notes and limitations

- Random seed: 42; CPU execution; compact neural models; two training epochs per comparison run and three for the final rerun.
- Forecast horizon: next available record for each station, not necessarily a fixed six-hour or daily horizon because sampling intervals may vary.
- Sequence history: 12 observations at the same station.
- Missing static numeric metadata (`latitude`, `longitude`, `wellDepth`) are median-imputed; rows missing target/time/group key are removed.
- Data collection is observational and irregular; results may differ by station and season. Future work should report station-wise error, fixed-horizon resampling, prediction intervals, and an untouched chronological test period.
- The five names refer to compact hybrid neural architectures implementing the requested combinations; `ConvLSTM + Attention` is implemented as temporal Conv1D + LSTM + attention pooling, a practical 1-D time-series analogue rather than a 2-D image ConvLSTM cell.

## 11. Deliverable files

- `before_preprocessing.csv`: untouched source rows and columns.
- `five_algorithms_dataset.csv`: cleaned rows and predictions from all five models.
- `three_algorithms_dataset.csv`: same rows and predictions from the top three.
- `two_algorithms_dataset.csv`: same rows and predictions from the top two.
- `final_algorithm_dataset.csv`: same rows and predictions from the final model.
- `feature_variance_importance.csv`: variance and importance diagnostic.
- `metrics_*.csv`: machine-readable metrics for each comparison stage.
