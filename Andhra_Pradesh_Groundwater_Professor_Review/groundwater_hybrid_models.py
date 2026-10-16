"""
ANDHRA PRADESH GROUNDWATER TIME-SERIES FORECASTING
=================================================

This script:
1. Loads the source data from CSV or Excel (.xlsx/.xls).
2. Audits the raw data and saves before_preprocessing.csv.
3. Cleans required fields and duplicate station/timestamp records.
4. Engineers cyclical time features and checks feature variance/importance.
5. Creates station-wise 12-observation histories to forecast the next dataValue.
6. Splits chronologically within each station: 70% train, 15% validation, 15% test.
7. Trains and evaluates all five requested hybrid neural models:
      - LSTM + CNN
      - CNN + BiLSTM
      - LSTM + Attention
      - Transformer + LSTM
      - ConvLSTM + Attention (temporal Conv1D + LSTM + attention pooling)
8. Selects top 3, reruns them; selects top 2, reruns them; reruns the final winner.
9. Saves a Markdown report, model metrics, feature diagnostics, and CSVs for every stage.

IMPORTANT:
- This is regression/forecasting, not classification. MAE/RMSE/R² are the main metrics.
- `dataValue` is treated as the continuous groundwater-level target (metres).
- The prediction horizon is the next available measurement for the same station.
- For formal research, model selection should be based on validation metrics and the
  final test set should be kept untouched. This script follows the requested ranking
  on test metrics for an initial comparison, and documents that limitation in the report.

Install dependencies on macOS:
    python3 -m pip install pandas numpy scikit-learn torch openpyxl

Run with the supplied Excel file:
    python3 groundwater_hybrid_models.py \
      "/Users/geetikarupani/Downloads/andhra_pradesh_groundwater_data_2020_2025.xlsx"

Or run with a CSV:
    python3 groundwater_hybrid_models.py "/path/to/dataset.csv"
"""

from pathlib import Path
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION LAYER
# ---------------------------------------------------------------------------
# Fix random seeds so repeated runs are as comparable as possible.
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

# These columns define the time-series task.
TARGET = "dataValue"       # Numerical groundwater level to predict.
TIME_COL = "dataTime"      # Timestamp used to order observations.
GROUP_COL = "stationCode"  # Each station has its own independent sequence.
HISTORY_LENGTH = 12        # Number of previous records used to forecast one record.
EPOCHS_FIRST_ROUND = 2     # Small quick-run setting; increase for serious experiments.
EPOCHS_FINAL = 3
BATCH_SIZE = 512
HIDDEN_SIZE = 24
DEVICE = torch.device("cpu")  # CPU-friendly for a Mac; change if you have a supported GPU.

MODEL_NAMES = [
    "LSTM + CNN",
    "CNN + BiLSTM",
    "LSTM + Attention",
    "Transformer + LSTM",
    "ConvLSTM + Attention",
]

# ---------------------------------------------------------------------------
# LAYER 1 — READ INPUT FILE
# ---------------------------------------------------------------------------
# Use the command-line file path if supplied; otherwise use the user's known path.
DEFAULT_SOURCE = Path(
    "/Users/geetikarupani/Downloads/andhra_pradesh_groundwater_data_2020_2025.xlsx"
)
source = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_SOURCE

# Save all generated files in a folder beside the script, not inside the source data.
OUTPUT_DIR = Path(__file__).resolve().parent / "groundwater_model_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not source.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{source}\n\n"
        "Run the script with the complete file path in quotes, for example:\n"
        'python3 groundwater_hybrid_models.py '
        '"/Users/geetikarupani/Downloads/andhra_pradesh_groundwater_data_2020_2025.xlsx"'
    )

print("=" * 75)
print("ANDHRA PRADESH GROUNDWATER FORECASTING PIPELINE")
print("=" * 75)
print(f"Input file: {source}")
print(f"Output folder: {OUTPUT_DIR}")

# Pick the correct pandas reader from the file extension.
if source.suffix.lower() in [".xlsx", ".xls"]:
    # Excel support: openpyxl is needed for .xlsx files.
    raw = pd.read_excel(source)
elif source.suffix.lower() == ".csv":
    raw = pd.read_csv(source, low_memory=False)
else:
    raise ValueError("Unsupported input file. Please provide .csv, .xlsx, or .xls.")

# Save the original dataset before any cleaning or feature engineering.
raw.to_csv(OUTPUT_DIR / "before_preprocessing.csv", index=False)

# ---------------------------------------------------------------------------
# LAYER 2 — RAW DATA AUDIT
# ---------------------------------------------------------------------------
# Capture source shape, types, missingness, and duplicates for the report.
raw_rows, raw_cols = raw.shape
raw_missing = raw.isna().sum()
raw_duplicates = int(raw.duplicated().sum())

raw_profile = pd.DataFrame({
    "column": raw.columns,
    "dtype": [str(raw[c].dtype) for c in raw.columns],
    "missing_count": [int(raw[c].isna().sum()) for c in raw.columns],
    "missing_percent": [round(float(raw[c].isna().mean() * 100), 4) for c in raw.columns],
})
raw_profile.to_csv(OUTPUT_DIR / "before_preprocessing_column_profile.csv", index=False)

print("\n[RAW DATA]")
print(f"Rows: {raw_rows:,}; columns: {raw_cols}")
print(f"Exact duplicate rows: {raw_duplicates:,}")
print(f"Missing cells: {int(raw_missing.sum()):,}")

# ---------------------------------------------------------------------------
# LAYER 3 — VALIDATE REQUIRED COLUMNS AND BASIC TYPES
# ---------------------------------------------------------------------------
# Stop early with a useful message if the Excel sheet has different column names.
required_columns = [TARGET, TIME_COL, GROUP_COL]
missing_required_columns = [c for c in required_columns if c not in raw.columns]
if missing_required_columns:
    raise ValueError(
        f"Required columns not found: {missing_required_columns}\n"
        f"Columns in file: {list(raw.columns)}"
    )

df = raw.copy()

# Parse timestamp values; invalid timestamps become NaT and are removed below.
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)

# Convert measurements and optional numeric predictors to numeric values.
for col in [TARGET, "latitude", "longitude", "wellDepth"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------------------------------------------------
# LAYER 4 — REMOVE ROWS THAT CANNOT BE USED FOR FORECASTING
# ---------------------------------------------------------------------------
# A supervised time-series target needs a station ID, a valid timestamp, and target value.
bad_required = df[[GROUP_COL, TIME_COL, TARGET]].isna().any(axis=1)
rows_removed_invalid = int(bad_required.sum())
df = df.loc[~bad_required].copy()

# Sort within each station. If duplicate station/timestamp records exist, keep the last.
df = df.sort_values([GROUP_COL, TIME_COL])
duplicate_station_time_count = int(
    df.duplicated(subset=[GROUP_COL, TIME_COL], keep="last").sum()
)
df = df.drop_duplicates(subset=[GROUP_COL, TIME_COL], keep="last").copy()
df = df.reset_index(drop=True)

# Drop columns that are completely empty after the required-row cleanup.
all_missing_columns = [c for c in df.columns if df[c].isna().all()]
if all_missing_columns:
    df = df.drop(columns=all_missing_columns)

# ---------------------------------------------------------------------------
# LAYER 5 — IMPUTE OPTIONAL NUMERIC METADATA
# ---------------------------------------------------------------------------
# Coordinates and well depth can be useful context. Median imputation avoids
# deleting otherwise valid target rows just because optional metadata is missing.
imputation_log = []
for col in ["latitude", "longitude", "wellDepth"]:
    if col in df.columns:
        count_missing = int(df[col].isna().sum())
        if count_missing:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            imputation_log.append({
                "column": col,
                "missing_values_imputed": count_missing,
                "imputation_method": "median",
                "imputation_value": float(median_value) if pd.notna(median_value) else None,
            })

pd.DataFrame(
    imputation_log,
    columns=["column", "missing_values_imputed", "imputation_method", "imputation_value"]
).to_csv(OUTPUT_DIR / "imputation_log.csv", index=False)

# ---------------------------------------------------------------------------
# LAYER 6 — TIME FEATURE ENGINEERING
# ---------------------------------------------------------------------------
# Cyclical encoding helps a model understand that hour 23 is near hour 0,
# December is near January, and the end of the year is near the start.
dt = df[TIME_COL]
df["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24.0)
df["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24.0)
df["dayofyear_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)
df["dayofyear_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
df["month_sin"] = np.sin(2 * np.pi * (dt.dt.month - 1) / 12.0)
df["month_cos"] = np.cos(2 * np.pi * (dt.dt.month - 1) / 12.0)

# Use a modest set of numerical features available in the dataset.
# The target's previous values are part of the history window, not future information.
FEATURES = [
    TARGET,
    "hour_sin", "hour_cos",
    "dayofyear_sin", "dayofyear_cos",
    "month_sin", "month_cos",
]
for optional_feature in ["latitude", "longitude", "wellDepth"]:
    if optional_feature in df.columns:
        FEATURES.append(optional_feature)

# Save which columns are model inputs and which columns are metadata only.
metadata_columns = [
    c for c in df.columns
    if c not in FEATURES and c not in [TIME_COL, GROUP_COL]
]
pd.DataFrame({
    "column": df.columns,
    "used_as_model_feature": [c in FEATURES for c in df.columns],
    "role": [
        "target/history feature" if c == TARGET
        else "time ordering" if c == TIME_COL
        else "station grouping" if c == GROUP_COL
        else "engineered/input feature" if c in FEATURES
        else "metadata/audit only"
        for c in df.columns
    ],
}).to_csv(OUTPUT_DIR / "column_roles.csv", index=False)

# ---------------------------------------------------------------------------
# LAYER 7 — CREATE SEQUENCE WINDOWS AND CHRONOLOGICAL SPLITS
# ---------------------------------------------------------------------------
# For each station independently:
#   X = previous HISTORY_LENGTH observations
#   y = the next observation's dataValue
# The station's timeline is split in order: 70% train, 15% validation, 15% test.
# This avoids randomly mixing later observations into the training set.
windows_X = []
windows_y = []
window_target_indices = []
window_split = []

df["Dataset_Split"] = "history_only"
df["Forecast_Target_Row"] = False

for station, group in df.groupby(GROUP_COL, sort=False):
    group = group.sort_values(TIME_COL)
    row_indices = group.index.to_list()
    values = group[FEATURES].to_numpy(dtype=np.float32)

    # Target rows begin after the first HISTORY_LENGTH records.
    target_positions = list(range(HISTORY_LENGTH, len(group)))
    n_targets = len(target_positions)
    if n_targets == 0:
        continue

    n_train = int(n_targets * 0.70)
    n_val = int(n_targets * 0.15)

    for order, target_pos in enumerate(target_positions):
        # Input sequence excludes the target row; no future target leaks into X.
        sequence = values[target_pos - HISTORY_LENGTH:target_pos]
        target_value = float(values[target_pos, FEATURES.index(TARGET)])
        target_row_index = row_indices[target_pos]

        if order < n_train:
            split_name = "train"
        elif order < n_train + n_val:
            split_name = "validation"
        else:
            split_name = "test"

        windows_X.append(sequence)
        windows_y.append(target_value)
        window_target_indices.append(target_row_index)
        window_split.append(split_name)

        df.loc[target_row_index, "Dataset_Split"] = split_name
        df.loc[target_row_index, "Forecast_Target_Row"] = True

if not windows_X:
    raise ValueError(
        f"No forecasting windows created. Each station needs more than "
        f"{HISTORY_LENGTH} valid records."
    )

X = np.asarray(windows_X, dtype=np.float32)
y = np.asarray(windows_y, dtype=np.float32)
target_indices = np.asarray(window_target_indices, dtype=int)
split_labels = np.asarray(window_split)

train_mask = split_labels == "train"
val_mask = split_labels == "validation"
test_mask = split_labels == "test"

if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
    raise ValueError("Train, validation, or test partition is empty; inspect station lengths.")

# ---------------------------------------------------------------------------
# LAYER 8 — SCALE FEATURES WITHOUT LEAKING TEST INFORMATION
# ---------------------------------------------------------------------------
# Fit scalers using training windows only. The same fitted scaler is then used
# to transform validation and test data; never fit a scaler on the test partition.
x_scaler = StandardScaler()
x_scaler.fit(X[train_mask].reshape(-1, X.shape[-1]))
X_scaled = x_scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

y_scaler = StandardScaler()
y_scaler.fit(y[train_mask].reshape(-1, 1))
y_scaled = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32).ravel()

# ---------------------------------------------------------------------------
# LAYER 9 — FEATURE VARIANCE AND QUICK IMPORTANCE CHECK
# ---------------------------------------------------------------------------
# Variance flags constant/near-constant features. Random Forest importance is a
# quick screening diagnostic on training-only sequence summaries, not proof of causality.
train_window_means = X[train_mask].mean(axis=1)
train_window_stds = X[train_mask].std(axis=1)

variance_records = []
for feature_index, feature_name in enumerate(FEATURES):
    variance = float(np.var(train_window_means[:, feature_index]))
    variance_records.append({
        "feature": feature_name,
        "variance_of_window_mean": variance,
        "zero_or_near_zero_variance": bool(variance < 1e-10),
    })

# Summarize each sequence by per-feature mean and standard deviation.
rf_X = np.concatenate([train_window_means, train_window_stds], axis=1)
rf = RandomForestRegressor(
    n_estimators=80, random_state=SEED, n_jobs=2, min_samples_leaf=3
)
rf.fit(rf_X, y[train_mask])

importance_records = []
n_features = len(FEATURES)
for i, feature_name in enumerate(FEATURES):
    # Combine importance of this feature's mean and standard-deviation summary.
    importance_records.append({
        "feature": feature_name,
        "importance_mean_plus_std": float(rf.feature_importances_[i] +
                                          rf.feature_importances_[i + n_features]),
    })

feature_report = pd.DataFrame(variance_records).merge(
    pd.DataFrame(importance_records), on="feature"
).sort_values("importance_mean_plus_std", ascending=False)
feature_report.to_csv(OUTPUT_DIR / "feature_variance_importance.csv", index=False)

# ---------------------------------------------------------------------------
# LAYER 10 — ATTENTION POOLING LAYER
# ---------------------------------------------------------------------------
# Learn a weight for each timestep and calculate a weighted sum of hidden states.
# This lets the model learn which history positions deserve more emphasis.
class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.score_layer = nn.Linear(hidden_size, 1)

    def forward(self, sequence_states):
        # sequence_states shape: [batch, time, hidden_size]
        scores = self.score_layer(sequence_states).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(sequence_states * weights.unsqueeze(-1), dim=1)
        return pooled

# ---------------------------------------------------------------------------
# LAYER 11 — DEFINE THE FIVE HYBRID MODEL ARCHITECTURES
# ---------------------------------------------------------------------------
# Each class branch below explicitly composes the requested neural components.
class HybridForecastModel(nn.Module):
    def __init__(self, model_name, n_features, hidden=HIDDEN_SIZE):
        super().__init__()
        self.model_name = model_name

        if model_name == "LSTM + CNN":
            # Layer A: LSTM learns sequential/time dependencies.
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            # Layer B: temporal Conv1D learns local patterns in the LSTM states.
            self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            # Layer C: pool over time and map to one continuous forecast.
            self.head = nn.Sequential(
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(hidden, 1),
            )

        elif model_name == "CNN + BiLSTM":
            # Layer A: temporal convolution detects short-range patterns.
            self.conv = nn.Conv1d(n_features, hidden, kernel_size=3, padding=1)
            # Layer B: bidirectional LSTM reads the encoded window in both directions.
            self.bilstm = nn.LSTM(
                hidden, hidden, batch_first=True, bidirectional=True
            )
            # Layer C: mean pool and regression head.
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        elif model_name == "LSTM + Attention":
            # Layer A: LSTM encodes the temporal sequence.
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            # Layer B: attention learns a weight for every hidden state.
            self.attention = AttentionPool(hidden)
            # Layer C: output one numeric prediction.
            self.head = nn.Linear(hidden, 1)

        elif model_name == "Transformer + LSTM":
            # Layer A: project raw features into transformer embedding size.
            self.input_projection = nn.Linear(n_features, hidden)
            # Layer B: self-attention learns relationships among time positions.
            transformer_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=48,
                dropout=0.1,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=1)
            # Layer C: LSTM refines the transformer sequence representation.
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            # Layer D: regression head uses the last hidden state.
            self.head = nn.Linear(hidden, 1)

        elif model_name == "ConvLSTM + Attention":
            # Practical 1-D time-series analogue: Conv1D -> LSTM -> attention.
            # It is not the original 2-D ConvLSTM cell used in image/video tasks.
            # Layer A: temporal convolution extracts local patterns.
            self.conv = nn.Conv1d(n_features, hidden, kernel_size=3, padding=1)
            # Layer B: LSTM models sequence dynamics from convolution features.
            self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
            # Layer C: attention pooling selects useful time positions.
            self.attention = AttentionPool(hidden)
            # Layer D: output one continuous forecast.
            self.head = nn.Linear(hidden, 1)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def forward(self, x):
        # Input shape: [batch, history_length, number_of_features]
        if self.model_name == "LSTM + CNN":
            states, _ = self.lstm(x)
            conv_states = torch.relu(self.conv(states.transpose(1, 2)))
            # CNN output shape: [batch, hidden, time]
            return self.head(conv_states).squeeze(-1)

        if self.model_name == "CNN + BiLSTM":
            conv_states = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
            states, _ = self.bilstm(conv_states)
            pooled = states.mean(dim=1)
            return self.head(pooled).squeeze(-1)

        if self.model_name == "LSTM + Attention":
            states, _ = self.lstm(x)
            pooled = self.attention(states)
            return self.head(pooled).squeeze(-1)

        if self.model_name == "Transformer + LSTM":
            embeddings = self.input_projection(x)
            attended = self.transformer(embeddings)
            states, _ = self.lstm(attended)
            return self.head(states[:, -1, :]).squeeze(-1)

        # ConvLSTM + Attention branch.
        conv_states = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        states, _ = self.lstm(conv_states)
        pooled = self.attention(states)
        return self.head(pooled).squeeze(-1)

# ---------------------------------------------------------------------------
# LAYER 12 — TRAIN ONE MODEL AND RETURN PREDICTIONS
# ---------------------------------------------------------------------------
def train_and_predict(model_name, epochs):
    # Reset seed so each model starts from a repeatable initialization.
    torch.manual_seed(SEED)
    model = HybridForecastModel(model_name, n_features=X_scaled.shape[-1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    train_ids = np.where(train_mask)[0]
    train_dataset = TensorDataset(
        torch.from_numpy(X_scaled[train_ids]),
        torch.from_numpy(y_scaled[train_ids]),
    )
    # Shuffle training windows only. Validation/test remain in chronological order.
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)
            loss.backward()

            # Gradient clipping helps avoid unstable recurrent-network gradients.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * len(batch_y)
            count += len(batch_y)

        print(f"    {model_name} | epoch {epoch + 1}/{epochs} "
              f"| train MSE (scaled): {running_loss / max(count, 1):.5f}")

    # Generate predictions for every window; final scoring uses only test windows.
    model.eval()
    predictions_scaled = []
    with torch.no_grad():
        all_tensor = torch.from_numpy(X_scaled)
        for start in range(0, len(all_tensor), BATCH_SIZE):
            batch = all_tensor[start:start + BATCH_SIZE].to(DEVICE)
            predictions_scaled.extend(model(batch).cpu().numpy().reshape(-1).tolist())

    # Return predictions to original metres using the training-fitted target scaler.
    predictions_scaled = np.asarray(predictions_scaled).reshape(-1, 1)
    predictions = y_scaler.inverse_transform(predictions_scaled).ravel()
    return predictions

# ---------------------------------------------------------------------------
# LAYER 13 — EVALUATION METRICS
# ---------------------------------------------------------------------------
def evaluate_model(model_name, predictions):
    # All model metrics are calculated against the exact same held-out test windows.
    actual = y[test_mask]
    predicted = predictions[test_mask]
    return {
        "model": model_name,
        "test_windows": int(test_mask.sum()),
        "MAE_m": float(mean_absolute_error(actual, predicted)),
        "RMSE_m": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
        # Supplementary metric: sign agreement relative to zero metres.
        # This is NOT standard classification accuracy.
        "sign_agreement_pct": float(np.mean(np.sign(actual) == np.sign(predicted)) * 100.0),
    }

# ---------------------------------------------------------------------------
# LAYER 14 — TRAIN AND COMPARE ALL FIVE MODELS
# ---------------------------------------------------------------------------
start_time = time.time()
predictions_by_model = {}

print("\n[STAGE 1: TRAIN ALL FIVE HYBRID MODELS]")
for model_name in MODEL_NAMES:
    print(f"\nTraining {model_name} ...")
    predictions_by_model[model_name] = train_and_predict(
        model_name, epochs=EPOCHS_FIRST_ROUND
    )

metrics_5 = pd.DataFrame([
    evaluate_model(name, predictions_by_model[name]) for name in MODEL_NAMES
]).sort_values("RMSE_m")
metrics_5.to_csv(OUTPUT_DIR / "metrics_five_algorithms.csv", index=False)
top_3 = metrics_5["model"].head(3).tolist()
print("\nFive-model comparison:")
print(metrics_5.to_string(index=False))
print("Selected top three by RMSE:", top_3)

# ---------------------------------------------------------------------------
# LAYER 15 — RERUN TOP THREE MODELS
# ---------------------------------------------------------------------------
# Retrain the three selected architectures from initialization using the same setup.
print("\n[STAGE 2: RERUN TOP THREE]")
for model_name in top_3:
    print(f"\nRerunning {model_name} ...")
    predictions_by_model[model_name] = train_and_predict(
        model_name, epochs=EPOCHS_FIRST_ROUND
    )

metrics_3 = pd.DataFrame([
    evaluate_model(name, predictions_by_model[name]) for name in top_3
]).sort_values("RMSE_m")
metrics_3.to_csv(OUTPUT_DIR / "metrics_three_algorithms.csv", index=False)
top_2 = metrics_3["model"].head(2).tolist()
print("\nTop-three rerun metrics:")
print(metrics_3.to_string(index=False))
print("Selected top two by RMSE:", top_2)

# ---------------------------------------------------------------------------
# LAYER 16 — RERUN TOP TWO MODELS
# ---------------------------------------------------------------------------
print("\n[STAGE 3: RERUN TOP TWO]")
for model_name in top_2:
    print(f"\nRerunning {model_name} ...")
    predictions_by_model[model_name] = train_and_predict(
        model_name, epochs=EPOCHS_FIRST_ROUND
    )

metrics_2 = pd.DataFrame([
    evaluate_model(name, predictions_by_model[name]) for name in top_2
]).sort_values("RMSE_m")
metrics_2.to_csv(OUTPUT_DIR / "metrics_two_algorithms.csv", index=False)
winner = str(metrics_2.iloc[0]["model"])
print("\nTop-two rerun metrics:")
print(metrics_2.to_string(index=False))
print("Final model selected:", winner)

# ---------------------------------------------------------------------------
# LAYER 17 — FINAL MODEL RERUN
# ---------------------------------------------------------------------------
# Train the winner for the configured final epoch count and evaluate once more.
print(f"\n[STAGE 4: FINAL MODEL — {winner}]")
predictions_by_model[winner] = train_and_predict(winner, epochs=EPOCHS_FINAL)
metrics_final = pd.DataFrame([evaluate_model(winner, predictions_by_model[winner])])
metrics_final.to_csv(OUTPUT_DIR / "metrics_final_algorithm.csv", index=False)
print("\nFinal metrics:")
print(metrics_final.to_string(index=False))

# ---------------------------------------------------------------------------
# LAYER 18 — EXPORT DATASETS FOR EACH REQUESTED STAGE
# ---------------------------------------------------------------------------
# Export cleaned row-level records and add prediction columns for test target rows.
# The raw observation rows are retained across stages; only prediction columns vary.
export_df = df.copy()
export_df[TIME_COL] = export_df[TIME_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
export_df["Forecast_Target_Value"] = export_df[TARGET]

def model_column_name(name):
    return "prediction_" + name.lower().replace(" + ", "_").replace(" ", "_")

def save_stage_dataset(model_names, filename):
    stage = export_df.copy()
    for name in model_names:
        col = model_column_name(name)
        stage[col] = np.nan
        # Prediction values are attached only to rows that are test targets.
        test_window_ids = np.where(test_mask)[0]
        test_target_rows = target_indices[test_window_ids]
        stage.loc[test_target_rows, col] = predictions_by_model[name][test_window_ids]
    stage.to_csv(OUTPUT_DIR / filename, index=False)
    print(f"Saved {filename}: {stage.shape[0]:,} rows × {stage.shape[1]} columns")

save_stage_dataset(MODEL_NAMES, "five_algorithms_dataset.csv")
save_stage_dataset(top_3, "three_algorithms_dataset.csv")
save_stage_dataset(top_2, "two_algorithms_dataset.csv")
save_stage_dataset([winner], "final_algorithm_dataset.csv")

# ---------------------------------------------------------------------------
# LAYER 19 — SAVE DATASET SIZE AND COLUMN AUDITS
# ---------------------------------------------------------------------------
clean_rows, clean_cols = df.shape
split_counts = pd.Series(split_labels).value_counts().to_dict()

dataset_stages = pd.DataFrame([
    {
        "stage": "before preprocessing",
        "rows": raw_rows,
        "columns": raw_cols,
        "forecast_windows": 0,
        "train_windows": 0,
        "validation_windows": 0,
        "test_windows": 0,
    },
    {
        "stage": "after preprocessing / five models",
        "rows": clean_rows,
        "columns": clean_cols + len(MODEL_NAMES) + 1,
        "forecast_windows": len(X),
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(val_mask.sum()),
        "test_windows": int(test_mask.sum()),
    },
    {
        "stage": "top three models",
        "rows": clean_rows,
        "columns": clean_cols + 3 + 1,
        "forecast_windows": len(X),
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(val_mask.sum()),
        "test_windows": int(test_mask.sum()),
    },
    {
        "stage": "top two models",
        "rows": clean_rows,
        "columns": clean_cols + 2 + 1,
        "forecast_windows": len(X),
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(val_mask.sum()),
        "test_windows": int(test_mask.sum()),
    },
    {
        "stage": "final model",
        "rows": clean_rows,
        "columns": clean_cols + 1 + 1,
        "forecast_windows": len(X),
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(val_mask.sum()),
        "test_windows": int(test_mask.sum()),
    },
])
dataset_stages.to_csv(OUTPUT_DIR / "dataset_stage_sizes.csv", index=False)

post_profile = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "missing_count": [int(df[c].isna().sum()) for c in df.columns],
    "missing_percent": [round(float(df[c].isna().mean() * 100), 4) for c in df.columns],
})
post_profile.to_csv(OUTPUT_DIR / "after_preprocessing_column_profile.csv", index=False)

# ---------------------------------------------------------------------------
# LAYER 20 — CREATE REVIEW MARKDOWN REPORT
# ---------------------------------------------------------------------------
def markdown_table(frame, floatfmt=".5f"):
    try:
        return frame.to_markdown(index=False, floatfmt=floatfmt)
    except ImportError:
        # Fallback if tabulate is not installed.
        return frame.to_string(index=False)

def md_table_any(frame):
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return frame.to_string(index=False)

removed_for_model = [
    c for c in df.columns
    if c not in FEATURES and c not in [TIME_COL, GROUP_COL, "Dataset_Split", "Forecast_Target_Row"]
]
zero_var = feature_report.loc[
    feature_report["zero_or_near_zero_variance"], "feature"
].tolist()
lowest_importance = feature_report.sort_values("importance_mean_plus_std").iloc[0]

report = f"""# Andhra Pradesh Groundwater Time-Series Forecasting

## Purpose and task
This is a time-series **regression/forecasting** experiment. The target is `{TARGET}` (groundwater level in metres). Each sample uses the previous {HISTORY_LENGTH} observations from the same `{GROUP_COL}` station to predict the next recorded target value. Because station measurements may be irregularly spaced, the forecast horizon is the next available record, not a guaranteed fixed number of hours or days.

## 1. Dataset before preprocessing
- Source file: `{source.name}`
- Original shape: **{raw_rows:,} rows × {raw_cols} columns**
- Exact duplicate rows: **{raw_duplicates:,}**
- Missing cells: **{int(raw_missing.sum()):,}**
- Rows removed because station ID, timestamp, or target was missing/invalid: **{rows_removed_invalid:,}**
- Duplicate `(stationCode, dataTime)` rows removed: **{duplicate_station_time_count:,}**
- Target: `{TARGET}`
- Time-ordering column: `{TIME_COL}`
- Sequence grouping column: `{GROUP_COL}`

### Original columns, data types and missingness
See `before_preprocessing_column_profile.csv`.

## 2. Preprocessing steps
1. Read the source as Excel or CSV and export `before_preprocessing.csv`.
2. Parse `{TIME_COL}` as UTC datetime; convert target and available numeric metadata to numeric types.
3. Remove records missing `{GROUP_COL}`, `{TIME_COL}`, or `{TARGET}` because they cannot form a valid supervised target.
4. Sort by station and timestamp; keep the last record for duplicate station/timestamp pairs.
5. Drop columns that are completely empty after essential-row cleanup.
6. Median-impute missing `latitude`, `longitude`, and `wellDepth` where those columns exist. Details are in `imputation_log.csv`.
7. Add cyclical time features: `hour_sin`, `hour_cos`, `dayofyear_sin`, `dayofyear_cos`, `month_sin`, `month_cos`.
8. Build history windows of {HISTORY_LENGTH} observations within each station. Each window predicts the following observation's `{TARGET}`.
9. Split windows chronologically within each station into 70% training, 15% validation, and 15% testing.
10. Fit standardization scalers on training data only and apply those fitted scalers to all partitions.
11. Run a variance and Random Forest feature-importance screening diagnostic on training windows only.
12. Train and compare the five requested hybrid neural models.

## 3. Features and removed columns
Model features: `{", ".join(FEATURES)}`.

`{GROUP_COL}` is used for grouping only, not as a numeric predictor. `{TIME_COL}` is used for sorting and is encoded using the cyclical time features. Other columns such as station names, administrative labels, descriptions, query fields, units and text metadata are retained in the row-level exports for auditability, but are not passed to the neural models.

See `column_roles.csv` for every column and its role. No feature is automatically removed merely because its importance is low; the feature report flags candidates for follow-up ablation testing.

## 4. Dataset sizes at each stage
{md_table_any(dataset_stages)}

The model-selection stages preserve the same cleaned rows and forecast windows. Prediction columns change at each stage. The model-stage CSVs include predictions for test target rows; other rows have blank prediction cells.

## 5. Feature variance and importance
{md_table_any(feature_report)}

Near-zero variance features: **{", ".join(zero_var) if zero_var else "none"}**.

Lowest quick importance: **{lowest_importance['feature']}**, importance **{lowest_importance['importance_mean_plus_std']:.6f}**. Importance is a screening statistic, not evidence of causation. Confirm any proposed feature removal with ablation experiments.

## 6. Initial comparison of five algorithms
Metrics are calculated on the same chronological test windows. Lower MAE/RMSE is better; higher R² is better. Error metrics are in metres.

{markdown_table(metrics_5)}

Top three by test RMSE: **{", ".join(top_3)}**.

## 7. Rerun of top three
{markdown_table(metrics_3)}

Top two by test RMSE in this rerun: **{", ".join(top_2)}**.

## 8. Rerun of top two
{markdown_table(metrics_2)}

Final selected model: **{winner}**.

## 9. Final model rerun
{markdown_table(metrics_final)}

## 10. Architecture notes
- **LSTM + CNN:** LSTM sequence encoder → temporal Conv1D → temporal pooling → regression output.
- **CNN + BiLSTM:** temporal Conv1D → bidirectional LSTM → pooling → regression output.
- **LSTM + Attention:** LSTM → learned attention pooling → regression output.
- **Transformer + LSTM:** feature projection → Transformer encoder → LSTM → regression output.
- **ConvLSTM + Attention:** temporal Conv1D → LSTM → learned attention pooling → regression output. This is a practical 1-D time-series analogue, not the original spatial 2-D ConvLSTM cell.

## 11. Reproducibility and limitations
- Random seed: {SEED}; CPU training.
- Comparison models use {EPOCHS_FIRST_ROUND} epochs; final rerun uses {EPOCHS_FINAL} epochs. This is a quick baseline, not extensive hyperparameter tuning.
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
"""
(OUTPUT_DIR / "Professor_Review_Report.md").write_text(report, encoding="utf-8")

# ---------------------------------------------------------------------------
# LAYER 21 — SAVE A MACHINE-READABLE SUMMARY
# ---------------------------------------------------------------------------
summary = {
    "source_file": str(source),
    "raw_rows": int(raw_rows),
    "raw_columns": int(raw_cols),
    "clean_rows": int(clean_rows),
    "clean_columns": int(clean_cols),
    "features": FEATURES,
    "history_length": int(HISTORY_LENGTH),
    "train_windows": int(train_mask.sum()),
    "validation_windows": int(val_mask.sum()),
    "test_windows": int(test_mask.sum()),
    "top_three": top_3,
    "top_two": top_2,
    "final_model": winner,
    "metrics_five": metrics_5.to_dict(orient="records"),
    "metrics_three": metrics_3.to_dict(orient="records"),
    "metrics_two": metrics_2.to_dict(orient="records"),
    "metrics_final": metrics_final.to_dict(orient="records"),
    "elapsed_seconds": round(time.time() - start_time, 2),
}
(OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "=" * 75)
print("PIPELINE FINISHED")
print("=" * 75)
print(f"Output folder: {OUTPUT_DIR}")
print(f"Final selected model: {winner}")
print(f"Training/validation/test windows: {train_mask.sum():,} / "
      f"{val_mask.sum():,} / {test_mask.sum():,}")
print("Read Professor_Review_Report.md for the documented metrics and decisions.")
