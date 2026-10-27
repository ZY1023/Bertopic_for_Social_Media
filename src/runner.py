"""Configuration-driven command line runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from importlib import metadata

from src.pipeline import load_config, prepare_documents


def file_hash(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BERTopic on a configured CSV dataset")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Validate data and tokenization without fitting")
    args = parser.parse_args()
    config = load_config(args.config)
    frame, checks = prepare_documents(config)
    if args.smoke:
        from src.modeling import make_tokenizer
        p = config["preprocessing"]
        tokenizer = make_tokenizer(p.get("chinese_tokenization", False), p.get("custom_dictionary"), p.get("stopwords"))
        sample = tokenizer(frame["model_text"].iloc[0])
        if not sample:
            raise ValueError("Tokenizer produced no tokens for first document")
        print(json.dumps({"preprocessing_checks": checks, "sample_tokens": sample[:20]}, ensure_ascii=False, indent=2))
        return
    cache = Path(config["dataset"]["output_dir"]) / ".numba_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))
    from src.modeling import fit_topics
    from src.reporting import write_outputs
    model, topics, probabilities = fit_topics(frame["model_text"].tolist(), config)
    versions = {}
    for package in ("bertopic", "sentence-transformers", "umap-learn", "hdbscan", "scikit-learn", "jieba", "pandas", "PyYAML"):
        versions[package] = metadata.version(package)
    output = write_outputs(model, frame, topics, probabilities, config, checks, versions, file_hash(config["dataset"]["input"]))
    print(json.dumps({"output": str(output), "modeled_documents": len(frame), "detected_topics": len(set(topics) - {-1}), "outlier_rate": sum(t == -1 for t in topics) / len(topics), "topic_sizes": {str(k): int(v) for k, v in frame.assign(topic_id=topics).groupby("topic_id").size().items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
