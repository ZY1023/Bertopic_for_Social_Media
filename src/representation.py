"""Representation-only Unicode cleanup, token audits, and vocabulary preview."""

from pathlib import Path
import re
import unicodedata

import jieba
from sklearn.feature_extraction.text import CountVectorizer


def normalize_representation_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    characters = []
    for char in text:
        category = unicodedata.category(char)
        if category == "Cf":  # BOM and zero-width formatting characters
            continue
        if char.isspace() or category == "Zs":
            characters.append(" ")
        else:
            characters.append(char)
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def strip_edge_marks(token: str) -> str:
    token = normalize_representation_text(token)
    left, right = 0, len(token)
    while left < right and unicodedata.category(token[left])[0] in "PSZC":
        left += 1
    while right > left and unicodedata.category(token[right - 1])[0] in "PSZC":
        right -= 1
    return token[left:right]


def token_issue(token: str) -> str:
    if not token or not token.strip():
        return "empty_or_whitespace"
    if all(unicodedata.category(char)[0] not in "LN" for char in token):
        return "punctuation_or_symbol_only"
    if token.casefold() in {"http", "https", "www", "com"}:
        return "platform_formatting"
    if token.isdecimal() and len(token) == 1:
        return "isolated_single_digit"
    return ""


class TopicTokenizer:
    """Pickle-safe jieba tokenizer for c-TF-IDF only."""

    def __init__(self, chinese: bool, excluded: set[str]):
        self.chinese = chinese
        self.excluded = excluded

    def inspect(self, text: str):
        raw = jieba.lcut(normalize_representation_text(text)) if self.chinese else normalize_representation_text(text).split()
        accepted, rejected = [], []
        for part in raw:
            token = strip_edge_marks(part)
            issue = token_issue(token)
            if issue == "empty_or_whitespace" and part.strip():
                issue = "punctuation_or_symbol_only"
            if not issue and token in self.excluded:
                issue = "stopword"
            if issue:
                rejected.append((part, issue))
            else:
                accepted.append(token)
        return raw, accepted, rejected

    def __call__(self, text: str) -> list[str]:
        return self.inspect(text)[1]


def make_tokenizer(chinese: bool, dictionary: str | None, stopwords: str | None):
    if dictionary:
        jieba.load_userdict(str(Path(dictionary)))
    excluded = set()
    if stopwords:
        excluded = {
            strip_edge_marks(line)
            for line in Path(stopwords).read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    return TopicTokenizer(chinese, excluded)


def make_vectorizer(config: dict) -> CountVectorizer:
    prep = config["preprocessing"]
    vec = config["vectorizer"].copy()
    vec["ngram_range"] = tuple(vec["ngram_range"])
    if prep.get("chinese_tokenization", False):
        vec.update(
            tokenizer=make_tokenizer(True, prep.get("custom_dictionary"), prep.get("stopwords")),
            token_pattern=None,
            lowercase=False,
        )
    return CountVectorizer(**vec)
