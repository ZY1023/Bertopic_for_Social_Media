"""Configurable BERTopic assembly."""

from src.representation import TopicTokenizer, make_tokenizer, make_vectorizer


def build_model(config: dict):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    model = config["model"]
    representation = model.get("representation", {"method": "ctfidf"})
    if representation["method"] == "ctfidf":
        representation_model = None
    elif representation["method"] == "keybert_inspired":
        from bertopic.representation import KeyBERTInspired
        representation_model = KeyBERTInspired(**representation.get("parameters", {}))
    else:
        raise ValueError("Unsupported representation method")
    hdbscan = model["hdbscan"].copy()
    hdbscan.setdefault("prediction_data", True)
    return BERTopic(
        embedding_model=SentenceTransformer(model["embedding_model"]),
        umap_model=UMAP(**model["umap"]),
        hdbscan_model=HDBSCAN(**hdbscan),
        vectorizer_model=make_vectorizer(config),
        top_n_words=model.get("top_n_words", 10),
        calculate_probabilities=model.get("calculate_probabilities", False),
        nr_topics=model.get("nr_topics"),
        representation_model=representation_model,
        verbose=True,
    )


def fit_topics(documents: list[str], config: dict):
    if not documents or any(not item.strip() for item in documents):
        raise ValueError("Every modeled document must have nonempty text")
    model = build_model(config)
    topics, probabilities = model.fit_transform(documents)
    return model, topics, probabilities
