"""Write generic BERTopic assignments and model artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_outputs(model, frame, topics, probabilities, config, checks, versions, fingerprint):
    dataset = config["dataset"]
    output = Path(dataset["output_dir"]) / dataset["name"]
    output.mkdir(parents=True, exist_ok=True)
    if len(topics) != len(frame):
        raise ValueError("Topic assignment count does not match modeled documents")

    columns = [dataset["id_column"], *dataset.get("metadata_columns", [])]
    assigned = frame[columns].copy()
    assigned.insert(1, "topic_id", topics)
    if probabilities is not None:
        values = np.asarray(probabilities)
        if values.ndim == 1 and len(values) == len(frame):
            assigned.insert(2, "assignment_confidence", values)
        elif values.ndim == 2 and len(values) == len(frame):
            assigned.insert(2, "assignment_confidence", values.max(axis=1))
    assigned.to_csv(output / "document_topics.csv", index=False, encoding="utf-8-sig")

    topic_info = model.get_topic_info()
    topic_info.to_csv(output / "topic_info.csv", index=False, encoding="utf-8-sig")
    if int(topic_info["Count"].sum()) != len(frame):
        raise ValueError("Topic counts do not reconcile with modeled documents")

    diagnostics = {
        "configuration": config,
        "software_versions": versions,
        "input_sha256": fingerprint,
        "corpus_size": len(frame),
        "preprocessing_checks": checks,
        "detected_topics": len(set(topics) - {-1}),
        "outlier_count": sum(topic == -1 for topic in topics),
        "serialization": "pickle; load only from trusted sources",
    }
    (output / "model_config.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    model_dir = output / "model"
    model_dir.mkdir(exist_ok=True)
    model.save(str(model_dir / "bertopic.pkl"), serialization="pickle", save_embedding_model=True)
    return output
