# BERTopic core

Fit a BERTopic model to a CSV file with one document per row. The code supports
Chinese tokenization with `jieba`; use `--no-chinese-tokenization` for other
languages. It does not include any research data or topic-specific vocabulary.

## Install

```bash
pip install -r requirements.txt
```

## Run

Prepare a UTF-8 CSV file with a `text` column, then run:

```bash
python run_analysis.py input.csv --text-column text --outdir outputs
```

The script saves `topics.csv`, `documents.csv`, and a fitted model under
`outputs/`. The first run downloads the selected embedding model.

Optional arguments include `--embedding-model`, `--min-topic-size`, and
`--no-chinese-tokenization`. See `python run_analysis.py --help`.
