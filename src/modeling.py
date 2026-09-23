"""Reusable BERTopic model setup for Chinese or multilingual text."""

from collections.abc import Sequence

import jieba
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


def tokenize(text: str) -> list[str]:
    """Split Chinese text with jieba and keep meaningful English tokens."""
    return [token.strip() for token in jieba.lcut(text) if len(token.strip()) > 1]


def build_model(
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    min_topic_size: int = 10,
    chinese_tokenization: bool = True,
) -> BERTopic:
    """Create a BERTopic model with configurable embeddings and clustering."""
    if min_topic_size < 2:
        raise ValueError("min_topic_size must be at least 2")

    vectorizer = (
        CountVectorizer(tokenizer=tokenize, token_pattern=None)
        if chinese_tokenization
        else CountVectorizer()
    )
    return BERTopic(
        embedding_model=SentenceTransformer(embedding_model),
        umap_model=UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, prediction_data=True),
        vectorizer_model=vectorizer,
    )


def fit_topics(documents: Sequence[str], **model_options) -> BERTopic:
    """Fit a model to nonempty texts and return it."""
    texts = [text.strip() for text in documents if isinstance(text, str) and text.strip()]
    if not texts:
        raise ValueError("No nonempty documents were provided")
    model = build_model(**model_options)
    model.fit(texts)
    return model
