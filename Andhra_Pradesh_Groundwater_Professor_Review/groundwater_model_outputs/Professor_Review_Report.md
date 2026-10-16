# Andhra Pradesh Groundwater Time-Series Forecasting

## Purpose and task
This is a time-series **regression/forecasting** experiment. The target is `dataValue` (groundwater level in metres). Each sample uses the previous 12 observations from the same `stationCode` station to predict the next recorded target value. Because station measurements may be irregularly spaced, the forecast horizon is the next available record, not a guaranteed fixed number of hours or days.

## 1. Dataset before preprocessing
- Source file: `andhra_pradesh_groundwater_data_2020_2025.xlsx`
- Original shape: **57,177 rows × 27 columns**
- Exact duplicate rows: **0**
- Missing cells: **264,813**
- Rows removed because station ID, timestamp, or target was missing/invalid: **0**
- Duplicate `(stationCode, dataTime)` rows removed: **1**
- Target: `dataValue`
- Time-ordering column: `dataTime`
- Sequence grouping column: `stationCode`

### Original columns, data types and missingness
See `before_preprocessing_column_profile.csv`.

## 2. Preprocessing steps
1. Read the source as Excel or CSV and export `before_preprocessing.csv`.
2. Parse `dataTime` as UTC datetime; convert target and available numeric metadata to numeric types.
3. Remove records missing `stationCode`, `dataTime`, or `dataValue` because they cannot form a valid supervised target.
4. Sort by station and timestamp; keep the last record for duplicate station/timestamp pairs.
5. Drop columns that are completely empty after essential-row cleanup.
6. Median-impute missing `latitude`, `longitude`, and `wellDepth` where those columns exist. Details are in `imputation_log.csv`.
7. Add cyclical time features: `hour_sin`, `hour_cos`, `dayofyear_sin`, `dayofyear_cos`, `month_sin`, `month_cos`.
8. Build history windows of 12 observations within each station. Each window predicts the following observation's `dataValue`.
9. Split windows chronologically within each station into 70% training, 15% validation, and 15% testing.
10. Fit standardization scalers on training data only and apply those fitted scalers to all partitions.
11. Run a variance and Random Forest feature-importance screening diagnostic on training windows only.
12. Train and compare the five requested hybrid neural models.

## 3. Features and removed columns
Model features: `dataValue, hour_sin, hour_cos, dayofyear_sin, dayofyear_cos, month_sin, month_cos, latitude, longitude, wellDepth`.

`stationCode` is used for grouping only, not as a numeric predictor. `dataTime` is used for sorting and is encoded using the cyclical time features. Other columns such as station names, administrative labels, descriptions, query fields, units and text metadata are retained in the row-level exports for auditability, but are not passed to the neural models.

See `column_roles.csv` for every column and its role. No feature is automatically removed merely because its importance is low; the feature report flags candidates for follow-up ablation testing.

## 4. Dataset sizes at each stage
| stage                             |   rows |   columns |   forecast_windows |   train_windows |   validation_windows |   test_windows |
|:----------------------------------|-------:|----------:|-------------------:|----------------:|---------------------:|---------------:|
| before preprocessing              |  57177 |        27 |                  0 |               0 |                    0 |              0 |
| after preprocessing / five models |  57176 |        39 |              55847 |           39051 |                 8337 |           8459 |
| top three models                  |  57176 |        37 |              55847 |           39051 |                 8337 |           8459 |
| top two models                    |  57176 |        36 |              55847 |           39051 |                 8337 |           8459 |
| final model                       |  57176 |        35 |              55847 |           39051 |                 8337 |           8459 |

The model-selection stages preserve the same cleaned rows and forecast windows. Prediction columns change at each stage. The model-stage CSVs include predictions for test target rows; other rows have blank prediction cells.

## 5. Feature variance and importance
| feature       |   variance_of_window_mean | zero_or_near_zero_variance   |   importance_mean_plus_std |
|:--------------|--------------------------:|:-----------------------------|---------------------------:|
| dataValue     |             160.29        | False                        |                0.9966      |
| latitude      |               2.09019     | False                        |                0.000716609 |
| dayofyear_cos |               0.422093    | False                        |                0.000713006 |
| dayofyear_sin |               0.545585    | False                        |                0.000669525 |
| longitude     |               3.61909     | False                        |                0.000542532 |
| hour_cos      |               0.000501093 | False                        |                0.00026844  |
| month_sin     |               0.53007     | False                        |                0.000167364 |
| month_cos     |               0.43235     | False                        |                0.00015641  |
| hour_sin      |               0.000466162 | False                        |                0.000119407 |
| wellDepth     |            5387.34        | False                        |                4.7097e-05  |

Near-zero variance features: **none**.

Lowest quick importance: **wellDepth**, importance **0.000047**. Importance is a screening statistic, not evidence of causation. Confirm any proposed feature removal with ablation experiments.

## 6. Initial comparison of five algorithms
Metrics are calculated on the same chronological test windows. Lower MAE/RMSE is better; higher R² is better. Error metrics are in metres.

| model                |   test_windows |   MAE_m |   RMSE_m |      R2 |   sign_agreement_pct |
|:---------------------|---------------:|--------:|---------:|--------:|---------------------:|
| LSTM + CNN           |           8459 | 1.06864 |  2.30724 | 0.96687 |             95.96879 |
| CNN + BiLSTM         |           8459 | 0.95127 |  2.77352 | 0.95213 |             95.21220 |
| ConvLSTM + Attention |           8459 | 1.20386 |  3.22473 | 0.93528 |             96.42984 |
| Transformer + LSTM   |           8459 | 1.10363 |  3.30716 | 0.93193 |             94.12460 |
| LSTM + Attention     |           8459 | 1.66152 |  3.82204 | 0.90909 |             94.12460 |

Top three by test RMSE: **LSTM + CNN, CNN + BiLSTM, ConvLSTM + Attention**.

## 7. Rerun of top three
| model                |   test_windows |   MAE_m |   RMSE_m |      R2 |   sign_agreement_pct |
|:---------------------|---------------:|--------:|---------:|--------:|---------------------:|
| LSTM + CNN           |           8459 | 1.06864 |  2.30724 | 0.96687 |             95.96879 |
| CNN + BiLSTM         |           8459 | 0.95127 |  2.77352 | 0.95213 |             95.21220 |
| ConvLSTM + Attention |           8459 | 1.20386 |  3.22473 | 0.93528 |             96.42984 |

Top two by test RMSE in this rerun: **LSTM + CNN, CNN + BiLSTM**.

## 8. Rerun of top two
| model        |   test_windows |   MAE_m |   RMSE_m |      R2 |   sign_agreement_pct |
|:-------------|---------------:|--------:|---------:|--------:|---------------------:|
| LSTM + CNN   |           8459 | 1.06864 |  2.30724 | 0.96687 |             95.96879 |
| CNN + BiLSTM |           8459 | 0.95127 |  2.77352 | 0.95213 |             95.21220 |

Final selected model: **LSTM + CNN**.

## 9. Final model rerun
| model      |   test_windows |   MAE_m |   RMSE_m |      R2 |   sign_agreement_pct |
|:-----------|---------------:|--------:|---------:|--------:|---------------------:|
| LSTM + CNN |           8459 | 0.86213 |  2.05749 | 0.97365 |             95.96879 |

## 10. Architecture notes
- **LSTM + CNN:** LSTM sequence encoder → temporal Conv1D → temporal pooling → regression output.
- **CNN + BiLSTM:** temporal Conv1D → bidirectional LSTM → pooling → regression output.
- **LSTM + Attention:** LSTM → learned attention pooling → regression output.
- **Transformer + LSTM:** feature projection → Transformer encoder → LSTM → regression output.
- **ConvLSTM + Attention:** temporal Conv1D → LSTM → learned attention pooling → regression output. This is a practical 1-D time-series analogue, not the original spatial 2-D ConvLSTM cell.

## 11. Reproducibility and limitations
- Random seed: 42; CPU training.
- Comparison models use 2 epochs; final rerun uses 3 epochs. This is a quick baseline, not extensive hyperparameter tuning.
- Model selection was requested to be performed using successive rankings. Since the same test partition is consulted repeatedly, the final test score is not an unbiased one-time estimate. For formal reporting, select models on validation metrics and reserve an untouched final chronological test period.
- The sign-agreement metric is supplementary and is not standard classification accuracy.
- Results may vary by station, season, sampling interval and data quality. Consider station-wise metrics, fixed-horizon resampling and prediction intervals for further research.

## 12. Output files
- `before_preprocessing.csv`
- `before_preprocessing_column_profile.csv`
- `after_preprocessing_column_profile.csv`
- `imputation_log.csv`
- `column_roles.csv`
- `dataset_stage_sizes.csv`
- `feature_variance_importance.csv`
- `five_algorithms_dataset.csv`
- `three_algorithms_dataset.csv`
- `two_algorithms_dataset.csv`
- `final_algorithm_dataset.csv`
- `metrics_five_algorithms.csv`
- `metrics_three_algorithms.csv`
- `metrics_two_algorithms.csv`
- `metrics_final_algorithm.csv`
