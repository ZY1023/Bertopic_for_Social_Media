import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.pipeline import clean_embedding_text, load_config, prepare_documents
from src.reporting import write_outputs
from src.configured_modeling import make_tokenizer
from src.representation import normalize_representation_text, make_vectorizer


def make_config(tmp_path):
    csv = tmp_path / "sample.csv"
    pd.DataFrame([
        {"doc": "a", "heading": "不应该删除中文", "body": None, "label": "yes", "likes": 10},
        {"doc": "b", "heading": None, "body": "自然语言处理很好", "label": "yes", "likes": 2},
        {"doc": "c", "heading": None, "body": None, "label": "yes", "likes": 3},
        {"doc": "d", "heading": "ignored", "body": "irrelevant", "label": "no", "likes": 4},
    ]).to_csv(csv, index=False)
    yaml = tmp_path / "config.yaml"
    yaml.write_text("""dataset:
  name: sample
  input: sample.csv
  encoding: utf-8
  id_column: doc
  text_columns: [heading, body]
  metadata_columns: [likes]
  filter: {column: label, accepted_values: ['yes']}
  output_dir: output
preprocessing: {language: zh, chinese_tokenization: true}
model:
  embedding_model: dummy
  umap: {random_state: 42}
  hdbscan: {min_cluster_size: 2}
vectorizer: {ngram_range: [1, 2], min_df: 1}
""", encoding="utf-8")
    return yaml


class FakeModel:
    def get_topic_info(self):
        return pd.DataFrame({"Topic": [0, -1], "Count": [1, 1]})

    def get_topic(self, topic):
        return [("语言", 0.6)] if topic == 0 else []

    def save(self, path, **kwargs):
        Path(path).mkdir()


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = make_config(self.root)

    def test_mapping_filter_missing_and_alignment(self):
        config = load_config(self.config_path)
        frame, checks = prepare_documents(config)
        self.assertEqual(frame["doc"].tolist(), ["a", "b"])
        self.assertEqual(frame["likes"].tolist(), [10, 2])
        self.assertEqual(frame["model_text"].tolist(), ["不应该删除中文", "自然语言处理很好"])
        self.assertEqual(len(frame["representation_text"]), len(frame["doc"]))
        self.assertEqual(checks, {"input_rows": 4, "after_filter": 3, "empty_after_filter": 1, "modeled_rows": 2})
        self.assertEqual(config, load_config(self.config_path))

    def test_single_column_and_chinese_preservation(self):
        config = load_config(self.config_path)
        config["dataset"]["text_columns"] = ["body"]
        frame, _ = prepare_documents(config)
        self.assertEqual(frame["doc"].tolist(), ["b"])
        self.assertEqual(clean_embedding_text("不要删除 中文 内容"), "不要删除 中文 内容")
        tokens = make_tokenizer(True, None, None)("不应该删除中文 内容")
        self.assertTrue(any("不" in token for token in tokens))
        self.assertTrue(any("中文" in token for token in tokens))
        stopwords = self.root / "stopwords.txt"
        stopwords.write_text("中文\n", encoding="utf-8")
        self.assertNotIn("中文", make_tokenizer(True, None, str(stopwords))("不应该删除中文 内容"))
        self.assertIn("中文", clean_embedding_text("不应该删除中文 内容"))

    def test_unicode_tokens_and_vectorizer(self):
        source = "\ufeff\u200b自然\u00a0语言 # 处理 🙂 不 NLP 70"
        normalized = normalize_representation_text(source)
        self.assertNotIn("\ufeff", normalized)
        self.assertNotIn("\u200b", normalized)
        self.assertNotIn("\u00a0", normalized)
        config = load_config(self.config_path)
        config["vectorizer"].update({"min_df": 2, "max_df": 0.85})
        vectorizer = make_vectorizer(config)
        self.assertEqual(vectorizer.ngram_range, (1, 2))
        self.assertIsNone(vectorizer.token_pattern)
        self.assertFalse(vectorizer.lowercase)
        tokens = vectorizer.tokenizer(normalized)
        self.assertIn("自然", tokens)
        self.assertIn("语言", tokens)
        self.assertIn("处理", tokens)
        self.assertIn("不", tokens)
        self.assertIn("70", tokens)
        self.assertNotIn("#", tokens)
        self.assertNotIn("🙂", tokens)
        self.assertTrue(all(any(char.isalnum() for char in token) for token in tokens))

    def test_output_reconciliation(self):
        config = load_config(self.config_path)
        frame, checks = prepare_documents(config)
        output = write_outputs(FakeModel(), frame, [0, -1], [0.8, 0.2], config, checks, {"test": "1"}, "abc")
        self.assertEqual(len(pd.read_csv(output / "document_topics.csv")), 2)
        self.assertEqual(pd.read_csv(output / "topic_info.csv")["Count"].sum(), 2)
        self.assertEqual(json.loads((output / "model_config.json").read_text(encoding="utf-8"))["corpus_size"], 2)
        with self.assertRaisesRegex(ValueError, "count"):
            write_outputs(FakeModel(), frame, [0], [0.8], config, checks, {}, "abc")


if __name__ == "__main__":
    unittest.main()
