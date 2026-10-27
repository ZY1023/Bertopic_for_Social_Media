"""CSV ingestion and pre-model checks."""

import re
from pathlib import Path

import pandas as pd
import yaml
from src.representation import normalize_representation_text


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for section in ("dataset", "preprocessing", "model", "vectorizer"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    dataset = config["dataset"]
    for key in ("name", "input", "id_column", "text_columns", "output_dir"):
        if key not in dataset:
            raise ValueError(f"Missing dataset setting: {key}")
    if isinstance(dataset["text_columns"], str):
        dataset["text_columns"] = [dataset["text_columns"]]
    if not dataset["text_columns"]:
        raise ValueError("text_columns must contain at least one column")
    if dataset.get("text_strategy", "join_nonempty") != "join_nonempty":
        raise ValueError("Unsupported text_strategy")
    base = path.resolve().parent
    file = Path(dataset["input"])
    dataset["input"] = str(file if file.is_absolute() else (base / file).resolve())
    output = Path(dataset["output_dir"])
    dataset["output_dir"] = str(output if output.is_absolute() else (base / output).resolve())
    for key in ("custom_dictionary", "stopwords"):
        value = config["preprocessing"].get(key)
        if value:
            file = Path(value)
            config["preprocessing"][key] = str(file if file.is_absolute() else (base / file).resolve())
    return config


def clean_embedding_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u200b", "")).strip()


def prepare_documents(config: dict):
    options = config["dataset"]
    frame = pd.read_csv(options["input"], encoding=options.get("encoding", "utf-8"), dtype={options["id_column"]: str})
    required = [options["id_column"], *options["text_columns"], *options.get("metadata_columns", [])]
    filter_spec = options.get("filter")
    if filter_spec:
        required.append(filter_spec["column"])
    absent = sorted(set(required) - set(frame.columns))
    if absent:
        raise ValueError(f"CSV is missing columns: {absent}")
    checks = {"input_rows": len(frame)}
    if filter_spec:
        frame = frame[frame[filter_spec["column"]].isin(filter_spec["accepted_values"])].copy()
    checks["after_filter"] = len(frame)
    separator = options.get("text_separator", "\n")
    texts = frame[options["text_columns"]].apply(
        lambda row: separator.join(part for part in (clean_embedding_text(value) for value in row) if part), axis=1
    )
    nonempty = texts.ne("")
    checks["empty_after_filter"] = int((~nonempty).sum())
    frame = frame.loc[nonempty].copy().reset_index(drop=True)
    frame["model_text"] = texts.loc[nonempty].tolist()
    frame["representation_text"] = frame["model_text"].map(normalize_representation_text)
    if frame[options["id_column"]].isna().any() or frame[options["id_column"]].duplicated().any():
        raise ValueError("Document IDs must be present and unique after filtering")
    checks["modeled_rows"] = len(frame)
    if not len(frame):
        raise ValueError("No documents remain after filtering")
    return frame, checks
