import unittest

from egy_tts_pipeline.config import GenerationConfig, NormalizationConfig
from egy_tts_pipeline.prompt_generation import PromptGenerator


class PromptGenerationTests(unittest.TestCase):
    def test_generate_requested_number_of_unique_rows(self) -> None:
        generator = PromptGenerator(GenerationConfig(seed=7), NormalizationConfig())
        rows = generator.generate(
            sample_ids=["sample-000001", "sample-000002", "sample-000003"],
            existing_texts=set(),
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(len({row["normalized_text"] for row in rows}), 3)
        self.assertTrue(all(not row["text_features"]["contains_latin"] for row in rows))
        self.assertTrue(all(not row["text_features"]["contains_digits"] for row in rows))
        self.assertTrue(all(3 <= row["text_features"]["word_count"] <= 18 for row in rows))


if __name__ == "__main__":
    unittest.main()
