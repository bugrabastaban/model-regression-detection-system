"""Offline structural and preservation checks for golden_v1."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest

from contracts import ClassifierOutput


DATASET = Path(__file__).resolve().parents[1] / "datasets" / "golden_v1.json"
# Canonical hash captured before the difficulty rename and draft additions.
ORIGINAL_CASES_SHA256 = "c93e8c56ea4a8a4d4246216cd8d100ee5037e30776aba32bc6489ca0b9594afe"


class GoldenDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATASET.read_text(encoding="utf-8-sig"))
        cls.cases = cls.dataset["cases"]

    def test_unique_ids_and_case_counts(self):
        ids = [case["id"] for case in self.cases]
        self.assertEqual(len(ids), 92)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [f"email_{i:03d}" for i in range(1, 93)])

    def test_required_fields_and_nonempty_text(self):
        fields = {"id", "input", "expected_category", "ideal_summary",
                  "expected_difficulty", "notes", "review_status"}
        self.assertEqual(self.dataset["dataset_version"], "golden_v1")
        for case in self.cases:
            with self.subTest(id=case.get("id")):
                self.assertEqual(set(case), fields)
                for field in fields:
                    self.assertIsInstance(case[field], str)
                    self.assertTrue(case[field].strip(), field)

    def test_valid_categories_and_output_contract(self):
        for case in self.cases:
            with self.subTest(id=case["id"]):
                self.assertIn(case["expected_category"], {"billing", "technical", "account", "general"})
                ClassifierOutput(category=case["expected_category"], summary=case["ideal_summary"])

    def test_difficulty_and_review_status(self):
        for case in self.cases:
            with self.subTest(id=case["id"]):
                self.assertNotIn("difficulty", case)
                self.assertIn(case["expected_difficulty"], {"easy", "medium", "hard"})
                self.assertIn(case["review_status"], {"approved", "pending"})
                # All 92 records, including the 12 edge cases, are now human-approved.
                self.assertEqual(case["review_status"], "approved")

    def test_original_80_preserved_including_difficulty_values(self):
        restored = [
            {("difficulty" if key == "expected_difficulty" else key): value
             for key, value in case.items()}
            for case in self.cases[:80]
        ]
        canonical = json.dumps(restored, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), ORIGINAL_CASES_SHA256)

    def test_draft_category_balance_and_documented_edge_types(self):
        drafts = self.cases[80:]
        self.assertEqual(Counter(c["expected_category"] for c in drafts),
                         {"billing": 3, "technical": 3, "account": 3, "general": 3})
        expected = {"ambiguous": 2, "extremely short": 3, "typos": 2,
                    "mixed-language": 3, "sarcastic": 2}
        for edge_type, count in expected.items():
            with self.subTest(edge_type=edge_type):
                selected = [c for c in drafts if c["notes"].startswith(f"Edge case: {edge_type}")]
                self.assertEqual(len(selected), count)
                if edge_type == "extremely short":
                    self.assertTrue(all(len(c["input"].split()) <= 6 for c in selected))


if __name__ == "__main__":
    unittest.main()
