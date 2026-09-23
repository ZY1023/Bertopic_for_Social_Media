"""Fit BERTopic to a CSV file containing one text document per row."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Input CSV file")
    parser.add_argument("--text-column", default="text", help="Column containing documents")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--no-chinese-tokenization", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    if args.text_column not in frame:
        parser.error(f"Column {args.text_column!r} was not found in {args.csv}")
    documents = frame[args.text_column].dropna().astype(str).str.strip()
    documents = documents[documents.ne("")].tolist()
    if len(documents) < args.min_topic_size:
        parser.error("The number of nonempty documents must reach --min-topic-size")

    from src.modeling import fit_topics

    model = fit_topics(
        documents,
        embedding_model=args.embedding_model,
        min_topic_size=args.min_topic_size,
        chinese_tokenization=not args.no_chinese_tokenization,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    model.get_topic_info().to_csv(args.outdir / "topics.csv", index=False)
    model.get_document_info(documents).to_csv(args.outdir / "documents.csv", index=False)
    model.save(args.outdir / "model", serialization="pickle")
    print(f"Saved topics, document assignments, and model to {args.outdir}")


if __name__ == "__main__":
    main()
