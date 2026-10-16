"""
Step-by-step preprocessing for Andhra Pradesh groundwater time-series data.

Target: dataValue (groundwater level, metres)
Timestamp: dataTime
Sequence/group key: stationCode

This script focuses on transparent preprocessing and dataset audit columns.
The accompanying ZIP also contains the results from the five trained hybrid models.
Run:
    python preprocess_and_audit.py "andhra_pradesh_groundwater_data_2020_2025 (1).csv"
Dependencies:
    pandas, numpy
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# STEP 1 — Load the original file and preserve a before-preprocessing copy.
source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/geetikarupani/Downloads/andhra_pradesh_groundwater_data_2020_2025.xlsx"
)
out_dir = Path("preprocessing_outputs")
out_dir.mkdir(exist_ok=True)

# Use the appropriate reader depending on the file extension.
if source.suffix.lower() in [".xlsx", ".xls"]:
    raw = pd.read_excel(source)
else:
    raw = pd.read_csv(source, low_memory=False)

print("STEP 1: Raw data")
print("Shape:", raw.shape)
print("Missing values by column:\n", raw.isna().sum().to_string())
print("Exact duplicate rows:", int(raw.duplicated().sum()))

# STEP 2 — Parse the time column and coerce numeric measurements.
df = raw.copy()
df["dataTime"] = pd.to_datetime(df["dataTime"], errors="coerce", utc=True)
for col in ["dataValue", "latitude", "longitude", "wellDepth"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# STEP 3 — Remove records that cannot be used for forecasting.
# A row needs a station identifier, valid timestamp, and observed target value.
required = ["stationCode", "dataTime", "dataValue"]
invalid_required = df[required].isna().any(axis=1)
print("Rows missing station/time/target:", int(invalid_required.sum()))
df = df.loc[~invalid_required].copy()

# STEP 4 — Sort chronologically within each station.
# If a station has multiple records at the same timestamp, keep the last one.
df = df.sort_values(["stationCode", "dataTime"])
duplicate_station_times = int(
    df.duplicated(subset=["stationCode", "dataTime"], keep="last").sum()
)
print("Duplicate station-time rows removed:", duplicate_station_times)
df = df.drop_duplicates(["stationCode", "dataTime"], keep="last").copy()

# STEP 5 — Fill missing numeric station metadata.
# These columns are auxiliary predictors, not the target.
for col in ["latitude", "longitude", "wellDepth"]:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# STEP 6 — Derive cyclical time features.
# Sine/cosine pairs preserve wrap-around (e.g. hour 23 is close to hour 0).
hour = df["dataTime"].dt.hour
day = df["dataTime"].dt.dayofyear
month = df["dataTime"].dt.month
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
df["dayofyear_sin"] = np.sin(2 * np.pi * day / 365.25)
df["dayofyear_cos"] = np.cos(2 * np.pi * day / 365.25)
df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

# STEP 7 — Document the features passed to the forecasting models.
# Raw text labels, IDs, and query bookkeeping are retained in the output CSV
# for auditability, but are not fed to the neural networks as numeric features.
model_features = [
    "dataValue", "hour_sin", "hour_cos", "dayofyear_sin",
    "dayofyear_cos", "month_sin", "month_cos",
]
for col in ["latitude", "longitude", "wellDepth"]:
    if col in df.columns:
        model_features.append(col)

# STEP 8 — Mark chronological forecasting roles per station.
# The first 12 observations form history only. Each later observation is a
# target predicted from the previous 12 observations of that same station.
#70-30 split for train-test, with 15% of the training set reserved for validation.
history_length = 12
df["Dataset_Split"] = "history_only"
df["Forecast_Target_Row"] = False
for station, group in df.groupby("stationCode", sort=False):
    group = group.sort_values("dataTime")
    target_rows = group.index.to_list()[history_length:]
    n = len(target_rows)
    n_train = int(n * 0.70)
    n_validation = int(n * 0.15)
    for i, row_index in enumerate(target_rows):
        if i < n_train:
            split = "train"
        elif i < n_train + n_validation:
            split = "validation"
        else:
            split = "test"
        df.loc[row_index, "Dataset_Split"] = split
        df.loc[row_index, "Forecast_Target_Row"] = True

# STEP 9 — Export the audit-ready preprocessed dataset and quality summary.
# This file includes all source columns plus engineered features and split labels.
df["dataTime"] = df["dataTime"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
df.to_csv(out_dir / "preprocessed_audit_dataset.csv", index=False)

quality = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing_count": [int(df[c].isna().sum()) for c in df.columns],
    "missing_percent": [float(df[c].isna().mean() * 100) for c in df.columns],
})
quality.to_csv(out_dir / "post_preprocessing_column_quality.csv", index=False)

print("\nSTEP 9: Preprocessed data")
print("Shape:", df.shape)
print("Model input features:", model_features)
print("Target: dataValue")
print("Split rows:\n", df["Dataset_Split"].value_counts().to_string())
print("Wrote outputs to:", out_dir.resolve())
