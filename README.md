# BERTopic topic modeling tool

Run BERTopic on a UTF-8 CSV containing one document per row. The project supports
configurable CSV columns, multilingual embeddings, optional Chinese word
segmentation with jieba, and topic assignment exports. No research data or
study-specific configuration is included.

## Install

```powershell
python -m pip install -r requirements.txt
```

The first model run may download the selected sentence embedding model.

## Input and configuration

Create a CSV with unique `id` values and nonempty `text` values. Copy
`configs/example.yaml` and set `dataset.input` to your CSV path. Paths in a
YAML file are resolved relative to that file. Change `id_column`,
`text_columns`, and optional `metadata_columns` to match your CSV.

The configuration controls the embedding model, UMAP, HDBSCAN, and
CountVectorizer settings. Set `chinese_tokenization: false` for non-Chinese
text. An optional custom jieba dictionary or stopword file can be supplied.

## Run

```powershell
python run_analysis.py --config configs/example.yaml --smoke
python run_analysis.py --config configs/example.yaml
```

The smoke command checks input mapping without fitting. The full command writes
`document_topics.csv`, `topic_info.csv`, `model_config.json`, and the fitted
model under the configured output directory. Generated outputs are ignored by
Git. Load pickle model files only from trusted sources.
