"""Generate preprocessed water-quality model-stage CSVs and accuracy summaries."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import (
    CategoryEmbeddingModelConfig,
    FTTransformerConfig,
    TabNetModelConfig,
    TabTransformerConfig,
)
from pytorch_tabular.models.danet.config import DANetConfig
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
SOURCE_CSV = ROOT / "Water_Quality_Dataset.csv"
if not SOURCE_CSV.exists():
    SOURCE_CSV = ROOT / "datasets" / "Water_Quality_Dataset.csv"
OUTPUT_DIR = ROOT / "model_stage_csvs"
TARGET = "Pollution_Level"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_EPOCHS = 3

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def get_model_stages():
    """Return the models to run at each comparison stage."""
    return {
        "five": [
            ("TabNet", TabNetModelConfig(task="classification")),
            ("FT-Transformer", FTTransformerConfig(task="classification")),
            ("TabTransformer", TabTransformerConfig(task="classification")),
            ("MLP", CategoryEmbeddingModelConfig(task="classification")),
            ("Deep & Cross Network", DANetConfig(task="classification")),
        ],
        "three": [
            ("FT-Transformer", FTTransformerConfig(task="classification")),
            ("MLP", CategoryEmbeddingModelConfig(task="classification")),
            ("TabTransformer", TabTransformerConfig(task="classification")),
        ],
        "two": [
            ("FT-Transformer", FTTransformerConfig(task="classification")),
            ("MLP", CategoryEmbeddingModelConfig(task="classification")),
        ],
        "final": [("FT-Transformer", FTTransformerConfig(task="classification"))],
    }


def feature_token(model_name: str) -> str:
    """Convert a display name to a stable, CSV-friendly column prefix."""
    return (
        model_name.lower()
        .replace("&", "and")
        .replace("-", "_")
        .replace(" ", "_")
    )


def prepare_dataset():
    """Load the raw CSV, engineer time features, and make a stratified split."""
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_CSV}")

    raw = pd.read_csv(SOURCE_CSV)
    if "Timestamp" not in raw.columns:
        raise ValueError("Dataset must contain a 'Timestamp' column.")
    if TARGET not in raw.columns:
        raise ValueError(f"Dataset must contain target column '{TARGET}'.")

    processed = raw.copy()
    timestamps = pd.to_datetime(
        processed["Timestamp"], dayfirst=True, errors="coerce"
    )
    if timestamps.isna().any():
        raise ValueError(
            f"Could not parse {int(timestamps.isna().sum())} Timestamp values."
        )

    processed["Hour"] = timestamps.dt.hour
    processed["DayOfWeek"] = timestamps.dt.dayofweek
    processed["Month"] = timestamps.dt.month
    processed = processed.drop(columns=["Timestamp"])

    categorical_columns = ["Location"]
    continuous_columns = [
        column
        for column in processed.columns
        if column not in categorical_columns + [TARGET]
    ]

    features = processed.drop(columns=[TARGET])
    target = processed[TARGET]
    train_features, validation_features, train_target, validation_target = (
        train_test_split(
            features,
            target,
            stratify=target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    train_data = train_features.copy()
    train_data[TARGET] = train_target
    validation_data = validation_features.copy()
    validation_data[TARGET] = validation_target

    export_data = processed.copy()
    export_data.insert(0, "Dataset_Split", "")
    export_data.loc[train_features.index, "Dataset_Split"] = "train"
    export_data.loc[
        validation_features.index, "Dataset_Split"
    ] = "validation_holdout"
    export_data = export_data.sort_index()

    return (
        raw,
        export_data,
        train_data,
        validation_data,
        validation_target,
        categorical_columns,
        continuous_columns,
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (
        raw,
        base_export,
        train_data,
        validation_data,
        validation_target,
        categorical_columns,
        continuous_columns,
    ) = prepare_dataset()

    shutil.copy2(SOURCE_CSV, OUTPUT_DIR / "before_preprocessing.csv")
    stage_scores = {}

    # Keep checkpoints and Lightning logs out of the repository.
    with tempfile.TemporaryDirectory(prefix="water_quality_model_runs_") as temp_dir:
        original_directory = Path.cwd()
        try:
            os.chdir(temp_dir)
            for stage, model_specs in get_model_stages().items():
                stage_export = base_export.copy()
                scores = []

                for model_name, model_config in model_specs:
                    model = TabularModel(
                        data_config=DataConfig(
                            target=[TARGET],
                            continuous_cols=continuous_columns,
                            categorical_cols=categorical_columns,
                            normalize_continuous_features=True,
                            continuous_feature_transform="quantile_normal",
                        ),
                        model_config=model_config,
                        optimizer_config=OptimizerConfig(optimizer="Adam"),
                        trainer_config=TrainerConfig(
                            max_epochs=MAX_EPOCHS,
                            accelerator="cpu",
                            devices=1,
                            check_val_every_n_epoch=1,
                            progress_bar=False,
                            deterministic=True,
                            seed=RANDOM_STATE,
                            checkpoints=None,
                            early_stopping=None,
                        ),
                        suppress_lightning_logger=True,
                        verbose=False,
                    )
                    model.fit(train_data, validation=validation_data)
                    predictions = model.predict(validation_data)

                    token = feature_token(model_name)
                    prediction_column = f"{token}_prediction"
                    stage_export[prediction_column] = pd.NA
                    stage_export.loc[
                        validation_data.index, prediction_column
                    ] = predictions["Pollution_Level_prediction"].to_numpy()

                    for class_id in sorted(validation_target.unique()):
                        probability_column = (
                            f"{token}_probability_class_{class_id}"
                        )
                        stage_export[probability_column] = pd.NA
                        stage_export.loc[
                            validation_data.index, probability_column
                        ] = predictions[
                            f"{TARGET}_{class_id}_probability"
                        ].to_numpy()

                    accuracy = accuracy_score(
                        validation_target,
                        predictions["Pollution_Level_prediction"],
                    )
                    scores.append((model_name, accuracy))

                stage_scores[stage] = scores
                stage_export.to_csv(
                    OUTPUT_DIR / f"{stage}_algorithms_dataset.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
        finally:
            os.chdir(original_directory)

    print(f"Raw dataset: {len(raw)} rows, {len(raw.columns)} columns")
    print(
        "Processed dataset: "
        f"{len(base_export)} rows, {len(base_export.columns)} columns "
        "(including Dataset_Split)"
    )
    print(
        f"Split: {len(train_data)} training rows, "
        f"{len(validation_data)} validation/holdout rows"
    )
    print(f"Target: {TARGET}")
    print(f"Output directory: {OUTPUT_DIR}")
    for stage, scores in stage_scores.items():
        formatted_scores = ", ".join(
            f"{name}={accuracy:.4f}" for name, accuracy in scores
        )
        print(f"{stage}: {formatted_scores}")


if __name__ == "__main__":
    main()
