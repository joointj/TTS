import unittest

from egy_tts_pipeline.config import NormalizationConfig
from egy_tts_pipeline.text_normalization import inspect_text_features, normalize_egyptian_text


class TextNormalizationTests(unittest.TestCase):
    def test_normalize_egyptian_text_removes_diacritics_and_normalizes_alef(self) -> None:
        config = NormalizationConfig()
        text = "إزيّك يا أَحمد؟  ده كِتابـي"
        self.assertEqual(normalize_egyptian_text(text, config), "ازيك يا احمد؟ ده كتابي")

    def test_inspect_text_features_finds_dialect_markers(self) -> None:
        features = inspect_text_features("أنا عايز أكلمك دلوقتي ومعلش استناني شوية")
        self.assertIn("عايز", features["dialect_markers"])
        self.assertIn("دلوقتي", features["dialect_markers"])
        self.assertFalse(features["contains_digits"])


if __name__ == "__main__":
    unittest.main()
